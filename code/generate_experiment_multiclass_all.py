import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

from tabarena.repository.evaluation_repository import load_repository

from utils import *
from ecemetrics import ERT
from ecemetrics.classifiers import *

from classifiers import InitLogitLGBMClassifier, MulticlassPartitionWisePredictor, WrapperNadarayaWatson, InitLogitXGBClassifier, InitLogitCatboostClassifier, IsotonicPredictor, SMSPredictor, TSPredictor, WSLGBMClassifier, WSCatboostClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from tabicl import TabICLClassifier
from tabpfn import TabPFNClassifier
from pathlib import Path
from calibration_generators import evaluate_ece_bin

import warnings
import os 

# warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", message="X does not have valid feature names")
# os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning'
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore::Warning'

import time

def prepare_dataset(repo, dataset, fold, config, cache):
    """Load and subsample dataset splits once, reuse for all calibrators."""
    key = (dataset, fold, config)
    if key in cache:
        return cache[key]

    p_cal = repo.predict_val(dataset=dataset, fold=fold, config=config, binary_as_multiclass=False)
    y_cal = repo.labels_val(dataset=dataset, fold=fold)
    
    if len(p_cal) >= 5_000:
        np.random.seed(123)
        idx = np.arange(0, len(p_cal))
        rand_idx = np.random.choice(idx, 5_000, replace=False)
        p_cal = p_cal[rand_idx]
        y_cal = y_cal[rand_idx]

    cache[key] = (p_cal, y_cal)
    return cache[key]

def main():
    parser = argparse.ArgumentParser(description="Script avec argument config_name")
    parser.add_argument("experiment_index", type=int, help="Index of the experiment (seed)")

    args = parser.parse_args()
    experiment_index = args.experiment_index  # Use this as seed
    
    repo = load_repository("D244_F3_C1530_100")
    configs = pd.read_csv('../logits/multiclass/configs.csv')

    data_cache = {}

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
                {"iterations": 300, "early_stopping_rounds": 200, "thread_count": 2, "random_state": experiment_index}
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
                MulticlassPartitionWisePredictor, 
                {"n_clusters": 30}
            ),
            (
                "NadarayaWatson", 
                WrapperNadarayaWatson, 
                {"kernel":'rbf', "h":0.1}
            ),
            (
                "WS_LGBMClassifier", 
                WSLGBMClassifier, 
                {}
            ),
            (
                "WS_CatboostClassifier", 
                WSCatboostClassifier, 
                {"iterations": 10, "early_stopping_rounds": 200, "thread_count": 2, "random_state": experiment_index}
            ),
            # (
            #     "InitLogitLGBMClassifier", 
            #     InitLogitLGBMClassifier, 
            #     {}
            # ),
            # (
            #     "InitLogitCatboostClassifier", 
            #     InitLogitCatboostClassifier, 
            #     {"iterations": 10, "early_stopping_rounds": 200, "thread_count": 2, "random_state": experiment_index}
            # ),
            # (
            #     "InitLogitXGBClassifier", 
            #     InitLogitXGBClassifier,
            #     {"n_estimators": 10, "random_state": experiment_index}
            # ),
            (
                "IsotonicPredictor", 
                IsotonicPredictor, 
                {}
            ),
            (
                "SMSPredictor", 
                SMSPredictor, 
                {}
            ),
            (
                "TSPredictor", 
                TSPredictor, 
                {}
            ),
        ]
    
    model_overfitting_configs = [
            (
                "IsotonicPredictor", 
                IsotonicPredictor, 
                {}
            ),
            (
                "SMSPredictor", 
                SMSPredictor, 
                {}
            ),
            (
                "TSPredictor", 
                TSPredictor, 
                {}
            ),
            (
                "PartitionWise", 
                MulticlassPartitionWisePredictor, 
                {"n_clusters": 15}
            ),
    ]

    print(f"\n=== Starting experiments ===")
    for _, row in tqdm(configs.iterrows(), total=len(configs)):
        seed_everything(experiment_index)
        experiment_data_list = []
        dataset, fold, config = row['dataset'], row['fold'], row['tuned_config']

        print("Dataset", dataset)
        print("fold", fold)
        print("config", config)
        try:
            p_test, y_test = prepare_dataset(
                repo, dataset, fold, config, data_cache
            )
        
            logits_test = multiclass_probs_to_logits(p_test)

            print("Number of test samples = ", len(y_test) )
            print("Number of classes = ", len(np.unique(y_test)) )

            if len(y_test) > 300:
                try:
                    print(f"\n=== Starting to calculate ECE ===")

                    current_row = {
                            "dataset": dataset,
                            "experiment_index": experiment_index,
                            "fold": fold,
                            "config": config,
                            "n_samples": len(y_test),
                            "n_classes": len(np.unique(y_test)),
                        }
                
                    for name, model_cls, kwargs in model_configs:
                        print(f"\n{name}")

                        start = time.time()
                        ert_model = ERT(model_cls, **kwargs)

                        eval_kwargs = {}
                        if name in ["InitLogitLGBMClassifier", "InitLogitXGBClassifier", "InitLogitCatboostClassifier"]:
                            eval_kwargs["init_logits"] = logits_test

                        metrics = ert_model.evaluate_multiple_losses(p_test, y_test, random_state=experiment_index, **eval_kwargs)
                            
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
                            eval_kwargs["init_logits"] = logits_test

                        metrics = ert_model.fit(p_test, y_test, **eval_kwargs)
                        metrics = ert_model.evaluate_multiple_losses(p_test, y_test, n_splits=1, **eval_kwargs)
                            
                        end = time.time()
                        elapsed_time = end - start
                        
                        metrics["time"] = elapsed_time
                        print(metrics)

                        current_row[f"time_overfitted_{name}"] = elapsed_time
                        for metric_name, metric_value in metrics.items():
                            if metric_name != "time":
                                current_row[f"{metric_name}_overfitted_{name}"] = metric_value

                    ece_l1 = evaluate_ece_bin(p_test, y_test, z=1, random_state=experiment_index)
                    ece_l2 = evaluate_ece_bin(p_test, y_test, z=2, random_state=experiment_index)
                    top_class_ECE = adaptive_top_class_ece(p_test, y_test)

                    current_row["ERT_L1_ECE_Binning_Classic"] = ece_l1
                    current_row["ERT_generalized_norm_score_2_Binning_Classic"] = ece_l2
                    current_row["top_class_ECE"] = top_class_ECE

                    file_path = Path("../results/multi_class_data_all_VF.csv")

                    df_new = pd.DataFrame([current_row])

                    id_cols = ["dataset", "experiment_index", "fold", "config"]
                    for col in id_cols:
                        df_new[col] = df_new[col].astype(str)
                    df_new.set_index(id_cols, inplace=True)
                    if file_path.exists():
                        df_existing = pd.read_csv(file_path, on_bad_lines='skip')
                        for col in id_cols:
                            df_existing[col] = df_existing[col].astype(str)
                        df_existing.set_index(id_cols, inplace=True)
                        df_final = df_new.combine_first(df_existing)
                    else:
                        df_final = df_new
                    df_final.reset_index().to_csv(file_path, index=False)



                except Exception as e:
                    print("ERROR FOR DATASET ", dataset)
                    print(f"An error occurred: {e}")
        except Exception as e:
            print("ERROR LOADING DATASET ", dataset)   
            print(f"An error occurred: {e}")
                
if __name__=="__main__":
    # import ray

    # if ray.is_initialized():
    #     ray.shutdown()
    # ray.init(num_cpus=1)

    # repo = load_repository("D244_F3_C1530_100")
    main()