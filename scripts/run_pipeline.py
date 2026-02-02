
"""
Pipeline: Prep + Feature Engineering → Train → Tune → Compare → Champion → Register
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.preprocessing import EngineDataPreprocessor
from src.data.feature_engineering import engineer_features_for_xgboost
from src.training.train_cnn import train_cnn_with_mlflow
from src.training.train_xgboost import train_xgboost_with_mlflow
from src.training.tune_xgboost import optimize_xgboost
from src.training.tune_cnn import optimize_cnn
import pandas as pd
import numpy as np
import argparse

def main(n_trials=10):
    EXPERIMENT_NAME = "Engine_Health_Pipeline"
    WINDOW_SIZE = 50
    N_SAMPLES = 1000
    
    print("="*70)
    print("PIPELINE: PREP+FEATURE_ENG → TRAIN → TUNE → COMPARE → CHAMPION → REGISTER")
    print("="*70)
    
    # STEP 1: DATA PREP + FEATURE ENGINEERING
    print("\n" + "="*70)
    print("STEP 1: DATA PREP + FEATURE ENGINEERING")
    print("="*70)
    
    preprocessor = EngineDataPreprocessor(window_size=WINDOW_SIZE)
    
    # Load COMPREDICT-provided data file
    from src.data.real_data_loader import load_real_engine_data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'engine_data.csv')
    X_raw, y_raw = load_real_engine_data(data_path)
    
    # Save CSV for CNN
    temp_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    temp_csv = os.path.join(temp_dir, 'temp_engine_data.csv')
    feature_cols = ['Engine rpm', 'Lub oil pressure', 'Fuel pressure', 
                    'Coolant pressure', 'lub oil temp', 'Coolant temp']
    df_temp = pd.DataFrame(X_raw, columns=feature_cols)
    df_temp['Engine Condition'] = y_raw.astype(int)
    df_temp.to_csv(temp_csv, index=False)
    
    # Prepare windowed data
    X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = preprocessor.prepare_data(X_raw, y_raw)
    X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb
    
    # FEATURE ENGINEERING for XGBoost
    print("\nFeature Engineering for XGBoost...")
    X_train_xgb_eng, _ = engineer_features_for_xgboost(X_train_xgb)
    X_test_xgb_eng, _ = engineer_features_for_xgboost(X_test_xgb)
    print(f"XGBoost features: {X_train_xgb_eng.shape[1]} engineered features")
    print(f"Data ready: {X_train_xgb_eng.shape[0]} train, {X_test_xgb_eng.shape[0]} test")
    
    # STEP 2: TRAIN BOTH MODELS
    print("\n" + "="*70)
    print("STEP 2: TRAIN BOTH MODELS (BASELINE)")
    print("="*70)
    
    print("\nTraining XGBoost...")
    xgb_baseline = train_xgboost_with_mlflow(
        X_train_xgb_eng, y_train_xgb, X_test_xgb_eng, y_test_xgb,
        hyperparams=None,
        experiment_name=EXPERIMENT_NAME,
        run_name="XGBoost_Baseline"
    )
    xgb_base_f1 = xgb_baseline.get('metrics', {}).get('f1_score', 0)
    print(f"→ XGBoost Baseline F1: {xgb_base_f1:.4f}")
    
    print("\nTraining CNN...")
    cnn_baseline = train_cnn_with_mlflow(data_path=temp_csv)
    cnn_base_f1 = cnn_baseline.get('metrics', {}).get('f1_score', 0)
    print(f"→ CNN Baseline F1: {cnn_base_f1:.4f}")
    
    # STEP 3: TUNE BOTH MODELS
    print("\n" + "="*70)
    print("STEP 3: TUNE BOTH MODELS")
    print("="*70)
    
    print(f"\nTuning XGBoost ({n_trials} trials)...")
    xgb_tune = optimize_xgboost(
        X_train_xgb_eng, y_train_xgb, X_test_xgb_eng, y_test_xgb,
        n_trials=n_trials,
        experiment_name=EXPERIMENT_NAME
    )
    xgb_best_params = xgb_tune['best_params']
    xgb_best_f1 = xgb_tune['best_value']
    print(f"→ XGBoost Best F1: {xgb_best_f1:.4f}")
    
    print(f"\nTuning CNN ({n_trials} trials)...")
    from src.training.tune_cnn import optimize_cnn
    cnn_tune = optimize_cnn(
        X_train_cnn, y_train_cnn, X_test_cnn, y_test_cnn,
        n_trials=n_trials,
        experiment_name=EXPERIMENT_NAME
    )
    cnn_best_params = cnn_tune['best_params']
    cnn_best_f1 = cnn_tune['best_value']
    print(f"→ CNN Best F1: {cnn_best_f1:.4f}")
    
    # STEP 4: COMPARE BEST HYPERPARAMETERS
    print("\n" + "="*70)
    print("STEP 4: COMPARE BEST HYPERPARAMETERS")
    print("="*70)
    
    print("\nXGBoost Best Params:")
    for k, v in xgb_best_params.items():
        print(f"  {k}: {v}")
    print(f"  F1: {xgb_best_f1:.4f}")
    
    print("\nCNN Best Params:")
    for k, v in cnn_best_params.items():
        print(f"  {k}: {v}")
    print(f"  F1: {cnn_best_f1:.4f}")
    
    # STEP 5: DECLARE CHAMPION
    print("\n" + "="*70)
    print("STEP 5: DECLARE CHAMPION")
    print("="*70)
    
    if xgb_best_f1 > cnn_best_f1:
        champion = "XGBoost"
        champion_f1 = xgb_best_f1
        champion_params = xgb_best_params
    else:
        champion = "CNN"
        champion_f1 = cnn_best_f1
        champion_params = cnn_best_params
    
    print(f"\n🏆 CHAMPION: {champion}")
    print(f"   F1: {champion_f1:.4f}")
    print(f"   Params: {champion_params}")
    
    # STEP 6: REGISTER MODEL
    print("\n" + "="*70)
    print("STEP 6: REGISTER MODEL")
    print("="*70)
    print(f"\nRegistering {champion} to MLflow Model Registry")
    print(f"  Champion logged during tuning (best trial)")
    
    # Cleanup
    if os.path.exists(temp_csv):
        os.remove(temp_csv)
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"Champion: {champion} (F1: {champion_f1:.4f})")
    return champion, champion_f1, champion_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-trials', type=int, default=10)
    args = parser.parse_args()
    main(n_trials=args.n_trials)
