import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

from utils import *
from ecemetrics import ERT
from ecemetrics.classifiers import *

from classifiers import InitLogitLGBMClassifier, PartitionWisePredictor, WrapperNadarayaWatson, InitLogitXGBClassifier, InitLogitCatboostClassifier, IsotonicPredictorBinary, TSPredictorBinary
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from tabicl import TabICLClassifier
from tabpfn import TabPFNClassifier
from pathlib import Path
from calibration_generators import *

import warnings
import os 

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")
os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning'

import time

def main():
    parser = argparse.ArgumentParser(description="Script avec argument config_name")
    parser.add_argument("experiment_index", type=int, help="Index of the experiment (seed)")

    args = parser.parse_args()
    experiment_index = args.experiment_index  # Use this as seed
    
    n_samples_max = 10_000
    tab_h_func = [perfectly_calibrated_bin, underconfident_binary, overconfident_binary, wavy_increasing_function_bin, almost_perfectly_calibrated_binary]

    if torch.backends.mps.is_available():  
        device = torch.device("mps")
    elif torch.cuda.is_available():        
        device = torch.device("cuda")
    else:                                  
        device = torch.device("cpu")
    print('Using device : ', device)

    model_configs = [
            (
                "BetterCatBoost", 
                BetterCatBoostClassifier, 
                {"iterations": 1000, "early_stopping_rounds": 200, "thread_count": 2, "random_state": experiment_index}
            ),
            (
                "CheapBetterLGBMClassifier", 
                CheapLGBMClassifier, 
                {}
            ),
            (
                "XT", 
                ExtraTreesClassifier, 
                {"n_estimators": 300, "max_depth": None, "random_state": experiment_index, "n_jobs": -1}
            ),
            (
                "RF", 
                RandomForestClassifier, 
                {"n_estimators": 300, "max_depth": None, "random_state": experiment_index, "n_jobs": -1}
            ),
            (
                "tabICLv2", 
                TabICLClassifier, 
                {}
            ),
            (
                "TabPFN", 
                TabPFNClassifier, 
                {"device": device, "ignore_pretraining_limits": True}
            ),
            (
                "PartitionWise", 
                PartitionWisePredictor, 
                {"n_clusters": 15}
            ),
            (
                "NadarayaWatson", 
                WrapperNadarayaWatson, 
                {"kernel":'rbf', "h":0.1}
            ),
            (
                "InitLogitLGBMClassifier", 
                InitLogitLGBMClassifier, 
                {}
            ),
            (
                "InitLogitCatboostClassifier", 
                InitLogitCatboostClassifier, 
                {"iterations": 10, "early_stopping_rounds": 200, "thread_count": 2, "random_state": experiment_index}
            ),
            # (
            #     "InitLogitXGBClassifier", 
            #     InitLogitXGBClassifier, 
            #     {"n_estimators": 10, "random_state": experiment_index}
            # ),
            (
                "IsotonicPredictor", 
                IsotonicPredictorBinary, 
                {}
            ),
            (
                "TSPredictor", 
                TSPredictorBinary, 
                {}
            ),
        ]
    
    model_overfitting_configs = [
            (
                "IsotonicPredictor", 
                IsotonicPredictorBinary, 
                {}
            ),
            (
                "TSPredictor", 
                TSPredictorBinary, 
                {}
            ),
            (
                "PartitionWise", 
                PartitionWisePredictor, 
                {"n_clusters": 15}
            ),
    ]

    print(f"\n=== Starting experiments ===")
    for function_pert in tab_h_func:
        experiment_data_list = [] 
        seed_everything(experiment_index)

        sim = BinaryCalibrationSimulator(h_func=function_pert, style="beta")
        true_L1_ECE = sim.calculate_true_L1_ece()

        values = np.geomspace(200, n_samples_max, num=15)
        tab_n_samples = np.round(values).astype(int)
        
        for n_samples in tab_n_samples:
            print("Starting with n samples = ", n_samples)
            try:
                preds = sim.generate_preds(n_samples=n_samples)
                logits = multiclass_probs_to_logits(preds)
                labels = sim.generate_labels(preds=preds)

                current_row = {
                        "h_func": function_pert.__name__,
                        "experiment_index": experiment_index,
                        "n_samples": n_samples,
                        "n_classes": "bin"
                    }
            
                for name, model_cls, kwargs in model_configs:
                    print(f"\n{name}")

                    start = time.time()
                    ert_model = ERT(model_cls, **kwargs)

                    eval_kwargs = {}
                    if name in ["InitLogitLGBMClassifier", "InitLogitXGBClassifier", "InitLogitCatboostClassifier"]:
                        eval_kwargs["init_logits"] = logits

                    metrics = ert_model.evaluate_multiple_losses(preds.reshape(-1, 1), labels, random_state=experiment_index, **eval_kwargs)
                        
                    end = time.time()
                    elapsed_time = end - start
                    
                    metrics["time"] = elapsed_time
                    print(metrics)

                    current_row[f"time_{name}"] = elapsed_time
                    for metric_name, metric_value in metrics.items():
                        if metric_name != "time":
                            current_row[f"{metric_name}_{name}"] = metric_value

                for name, model_cls, kwargs in model_overfitting_configs:
                    print(f"\n{name}")

                    start = time.time()
                    ert_model = ERT(model_cls, **kwargs)

                    eval_kwargs = {}
                    if name in ["InitLogitLGBMClassifier", "InitLogitXGBClassifier", "InitLogitCatboostClassifier"]:
                        eval_kwargs["init_logits"] = logits

                    metrics = ert_model.fit(preds.reshape(-1, 1), labels, **eval_kwargs)
                    metrics = ert_model.evaluate_multiple_losses(preds.reshape(-1, 1), labels, n_splits=1, **eval_kwargs)
                        
                    end = time.time()
                    elapsed_time = end - start
                    
                    metrics["time"] = elapsed_time
                    print(metrics)

                    current_row[f"time_overfitted_{name}"] = elapsed_time
                    for metric_name, metric_value in metrics.items():
                        if metric_name != "time":
                            current_row[f"{metric_name}_overfitted_{name}"] = metric_value

                ece_l1 = evaluate_ece_bin_1d(preds, labels)

                current_row["ERT_L1_ECE_Binning_Classic"] = ece_l1
                current_row["ERT_L1_ECE_TRUE"] = true_L1_ECE
                
                experiment_data_list.append(current_row)
                df_results = pd.DataFrame(experiment_data_list)
                file_path = Path("../results/synthetic_binary_V6.csv")

                id_cols = ["h_func", "experiment_index", "n_samples", "n_classes"]

                if file_path.exists():
                    df_existing = pd.read_csv(file_path)
                    df_existing.set_index(id_cols, inplace=True)
                    df_results.set_index(id_cols, inplace=True)
                    df_final = df_results.combine_first(df_existing).reset_index()
                else:
                    df_final = df_results
                df_final.to_csv(file_path, index=False)
            except Exception as e:
                print(f"An error occurred: {e}")

if __name__=="__main__":
    main()