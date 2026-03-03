import numpy as np
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import OneHotEncoder

import numpy as np
import sklearn
import multiprocessing as mp
import pandas as pd
import torch

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool
from probmetrics.calibrators import get_calibrator

from sklearn.metrics.pairwise import pairwise_kernels

class WrapperNadarayaWatson(BaseEstimator, ClassifierMixin):
    def __init__(self, h=0.1, kernel='rbf'):
        self.h = h
        self.kernel = kernel

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.X_train_ = X
        self.classes_ = np.unique(y)
        
        self.encoder_ = OneHotEncoder(sparse_output=False)
        self.y_train_one_hot_ = self.encoder_.fit_transform(y.reshape(-1, 1))
        
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        
        gamma = 0.5 / (self.h ** 2)
        K = pairwise_kernels(X, self.X_train_, metric=self.kernel, gamma=gamma)
        
        numerator = K @ self.y_train_one_hot_
        denominator = K.sum(axis=1)[:, np.newaxis]
        
        probs = numerator / np.maximum(denominator, 1e-12)
        
        row_sums = probs.sum(axis=1)
        probs[row_sums == 0] = 1.0 / len(self.classes_)
        
        return probs

    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]

class CheapBetterCatBoostClassifier(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    def _fit_model(self, idxs):
        m = CatBoostClassifier(iterations=1_000, learning_rate=0.08, subsample=0.9, bootstrap_type='Bernoulli',
                               max_depth=7, l2_leaf_reg=1e-5, random_strength=0.8, one_hot_max_size=15,
                               random_state=0, early_stopping_rounds=100, thread_count=1, verbose=0 if not self.verbose_ else 1, classes_count=len(self.classes_))
        return m.fit(self.X_.take(idxs[0], 0), self.y_[idxs[0]], eval_set=(self.X_.take(idxs[1], 0), self.y_[idxs[1]]))
    
    def fit(self, X, y, verbose=False, init_logits=None):
        if isinstance(X, torch.Tensor) or isinstance(X, pd.DataFrame) or isinstance(X, pd.Series): X = np.asarray(X)
        if isinstance(y, torch.Tensor) or isinstance(y, pd.DataFrame) or isinstance(y, pd.Series): y = np.asarray(y)
        self.verbose_ = verbose
        self.le_ = sklearn.preprocessing.LabelEncoder().fit(y)
        self.X_, self.y_, self.classes_ = X, self.le_.transform(y), self.le_.classes_
        splits = list(sklearn.model_selection.StratifiedKFold(n_splits=8, shuffle=True, random_state=0).split(X, y))
        with mp.Pool(processes=min(len(splits), mp.cpu_count())) as pool:
            self.models_ = pool.map(self._fit_model, splits)
        oof_preds = np.concatenate([m.predict_proba(X.take(idxs[1], 0)) for m, idxs in zip(self.models_, splits)],
                                   axis=0)
        oof_labels = np.concatenate([y[idxs[1]] for idxs in splits], axis=0)
        self.calib_ = get_calibrator('logistic', calibrate_with_mixture=True,
                                     logistic_binary_type='quadratic').fit(oof_preds, oof_labels)
        return self

    def predict_proba(self, X):
        if isinstance(X, torch.Tensor) or isinstance(X, pd.DataFrame) or isinstance(X, pd.Series): X = np.asarray(X)
        return self.calib_.predict_proba(np.mean([m.predict_proba(X) for m in self.models_], axis=0))

    def predict(self, X):
        if isinstance(X, torch.Tensor) or isinstance(X, pd.DataFrame) or isinstance(X, pd.Series): X = np.asarray(X)
        return self.le_.inverse_transform(np.argmax(self.predict_proba(X), axis=1))

class BetterLGBMClassifier(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    def _fit_model(self, idxs):
        n_classes = len(self.classes_)
        
        if n_classes <= 2:
            params = {
                "objective": "binary",
                "num_class": 1
            }
        else:
            params = {
                "objective": "multiclass",
                "num_class": n_classes
            }

        m = LGBMClassifier(n_estimators=10_000, learning_rate=0.02, subsample=0.75, subsample_freq=1, num_leaves=50,
                           random_state=0, early_stopping_round=300, min_child_samples=40, min_child_weight=1e-7,
                           n_jobs=1, verbosity=0 if not self.verbose_ else 1, **params)
        return m.fit(self.X_.take(idxs[0], 0), self.y_[idxs[0]], eval_set=(self.X_.take(idxs[1], 0), self.y_[idxs[1]]))

    def fit(self, X, y, verbose=False):
        if isinstance(X, torch.Tensor) or isinstance(X, pd.DataFrame) or isinstance(X, pd.Series): X = np.asarray(X)
        if isinstance(y, torch.Tensor) or isinstance(y, pd.DataFrame) or isinstance(y, pd.Series): y = np.asarray(y)
        self.verbose_ = verbose
        self.le_ = sklearn.preprocessing.LabelEncoder().fit(y)
        self.X_, self.y_, self.classes_ = X, self.le_.transform(y), self.le_.classes_
        splits = list(sklearn.model_selection.StratifiedKFold(n_splits=8, shuffle=True, random_state=0).split(X, y))
        with mp.Pool(processes=min(len(splits), mp.cpu_count())) as pool:
            self.models_ = pool.map(self._fit_model, splits)
        oof_preds = np.concatenate([m.predict_proba(X.take(idxs[1], 0)) for m, idxs in zip(self.models_, splits)],
                                   axis=0)
        oof_labels = np.concatenate([y[idxs[1]] for idxs in splits], axis=0)
        self.calib_ = get_calibrator('logistic', calibrate_with_mixture=True).fit(oof_preds, oof_labels)
        return self

    def predict_proba(self, X):
        if isinstance(X, torch.Tensor) or isinstance(X, pd.DataFrame) or isinstance(X, pd.Series): X = np.asarray(X)
        return self.calib_.predict_proba(np.mean([m.predict_proba(X) for m in self.models_], axis=0))

    def predict(self, X):
        if isinstance(X, torch.Tensor) or isinstance(X, pd.DataFrame) or isinstance(X, pd.Series): X = np.asarray(X)
        return self.le_.inverse_transform(np.argmax(self.predict_proba(X), axis=1))

class BetterCatBoostClassifier(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    def __init__(self, iterations=1000, early_stopping_rounds=300, thread_count=1, random_state=0):
        self.iterations = iterations
        self.early_stopping_rounds = early_stopping_rounds
        self.thread_count = thread_count
        self.random_state = random_state

    def _fit_model(self, idxs):
        m = CatBoostClassifier(
            iterations=self.iterations,
            learning_rate=None,
            random_state=self.random_state,
            early_stopping_rounds=self.early_stopping_rounds,
            thread_count=self.thread_count,
            classes_count=len(self.classes_),
            verbose=0 if not self.verbose_ else 1
        )
        return m.fit(
            self.X_.take(idxs[0], 0),
            self.y_[idxs[0]],
            eval_set=(self.X_.take(idxs[1], 0), self.y_[idxs[1]])
        )

    def fit(self, X, y, verbose=False):
        if isinstance(X, torch.Tensor) or isinstance(X, pd.DataFrame) or isinstance(X, pd.Series): X = np.asarray(X)
        if isinstance(y, torch.Tensor) or isinstance(y, pd.DataFrame) or isinstance(y, pd.Series): y = np.asarray(y)
        self.verbose_ = verbose
        self.le_ = sklearn.preprocessing.LabelEncoder().fit(y)
        self.X_, self.y_, self.classes_ = X, self.le_.transform(y), self.le_.classes_
        splits = list(
            sklearn.model_selection.StratifiedKFold(
                n_splits=8, shuffle=True, random_state=0
            ).split(X, y)
        )
        with mp.Pool(processes=min(len(splits), mp.cpu_count())) as pool:
            self.models_ = pool.map(self._fit_model, splits)
        oof_preds = np.concatenate([m.predict_proba(X.take(idxs[1], 0)) for m, idxs in zip(self.models_, splits)],
                                   axis=0)
        oof_labels = np.concatenate([y[idxs[1]] for idxs in splits], axis=0)
        self.calib_ = get_calibrator('logistic', calibrate_with_mixture=True,
                                     logistic_binary_type='quadratic').fit(oof_preds, oof_labels)
        return self

    def predict_proba(self, X):
        if isinstance(X, torch.Tensor) or isinstance(X, pd.DataFrame) or isinstance(X, pd.Series): X = np.asarray(X)
        return self.calib_.predict_proba(np.mean([m.predict_proba(X) for m in self.models_], axis=0))

    def predict(self, X):
        if isinstance(X, torch.Tensor) or isinstance(X, pd.DataFrame) or isinstance(X, pd.Series): X = np.asarray(X)
        return self.le_.inverse_transform(np.argmax(self.predict_proba(X), axis=1))

class CheapLGBMClassifier(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    def _fit_model(self, idxs):
        n_classes = len(self.classes_)
        
        if n_classes <= 2:
            params = {
                "objective": "binary",
                "num_class": 1
            }
        else:
            params = {
                "objective": "multiclass",
                "num_class": n_classes
            }

        m = LGBMClassifier(n_estimators=1_000, learning_rate=0.04, subsample=0.75, subsample_freq=1, num_leaves=50,
                           random_state=0, early_stopping_round=100, min_child_samples=40, min_child_weight=1e-7,
                           n_jobs=1, verbosity=self.verbose_, **params)
        return m.fit(self.X_.take(idxs[0], 0), self.y_[idxs[0]], eval_set=(self.X_.take(idxs[1], 0), self.y_[idxs[1]]))

    def fit(self, X, y, verbose=-1):
        if isinstance(X, torch.Tensor) or isinstance(X, pd.DataFrame) or isinstance(X, pd.Series): X = np.asarray(X)
        if isinstance(y, torch.Tensor) or isinstance(y, pd.DataFrame) or isinstance(y, pd.Series): y = np.asarray(y)
        self.verbose_ = verbose
        self.le_ = sklearn.preprocessing.LabelEncoder().fit(y)
        self.X_, self.y_, self.classes_ = X, self.le_.transform(y), self.le_.classes_
        splits = list(sklearn.model_selection.StratifiedKFold(n_splits=8, shuffle=True, random_state=0).split(X, y))
        with mp.Pool(processes=min(len(splits), mp.cpu_count())) as pool:
            self.models_ = pool.map(self._fit_model, splits)
        oof_preds = np.concatenate([m.predict_proba(X.take(idxs[1], 0)) for m, idxs in zip(self.models_, splits)],
                                   axis=0)
        oof_labels = np.concatenate([y[idxs[1]] for idxs in splits], axis=0)
        self.calib_ = get_calibrator('logistic', calibrate_with_mixture=True,
                                     logistic_binary_type='quadratic').fit(oof_preds, oof_labels)
        return self

    def predict_proba(self, X):
        if isinstance(X, torch.Tensor) or isinstance(X, pd.DataFrame) or isinstance(X, pd.Series): X = np.asarray(X)
        return self.calib_.predict_proba(np.mean([m.predict_proba(X) for m in self.models_], axis=0))

    def predict(self, X):
        if isinstance(X, torch.Tensor) or isinstance(X, pd.DataFrame) or isinstance(X, pd.Series): X = np.asarray(X)
        return self.le_.inverse_transform(np.argmax(self.predict_proba(X), axis=1))

import numpy as np
import pandas as pd
import torch
import multiprocessing as mp
import sklearn.base
import sklearn.model_selection
from catboost import CatBoostRegressor

class MyCatBoostRegressor(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    def __init__(self, iterations=10, early_stopping_rounds=300, thread_count=1, boost_from_average=True, random_state=0):
        self.iterations = iterations
        self.early_stopping_rounds = early_stopping_rounds
        self.thread_count = thread_count
        self.random_state = random_state
        self.boost_from_average = boost_from_average

    def _fit_model(self, idxs):
        m = CatBoostRegressor(
            iterations=self.iterations,
            learning_rate=None,
            random_state=self.random_state,
            early_stopping_rounds=self.early_stopping_rounds,
            thread_count=self.thread_count,
            verbose=0 if not self.verbose_ else 1,
            loss_function=self.loss_function_,
            boost_from_average=self.boost_from_average # Pass it here
        )
        return m.fit(
            self.X_.take(idxs[0], 0),
            self.y_[idxs[0]],
            eval_set=(self.X_.take(idxs[1], 0), self.y_[idxs[1]])
        )

    def fit(self, X, y, verbose=False):
        if isinstance(X, torch.Tensor) or isinstance(X, pd.DataFrame) or isinstance(X, pd.Series): X = np.asarray(X)
        if isinstance(y, torch.Tensor) or isinstance(y, pd.DataFrame) or isinstance(y, pd.Series): y = np.asarray(y)
        self.verbose_ = verbose
        self.X_, self.y_ = X, y
        if y.ndim > 1 and y.shape[1] > 1:
            self.loss_function_ = 'MultiRMSE'
        else:
            self.loss_function_ = 'RMSE'
        splits = list(
            sklearn.model_selection.KFold(
                n_splits=8, shuffle=True, random_state=0
            ).split(X, y)
        )
        with mp.Pool(processes=min(len(splits), mp.cpu_count())) as pool:
            self.models_ = pool.map(self._fit_model, splits)
        return self

    def predict(self, X):
        if isinstance(X, torch.Tensor) or isinstance(X, pd.DataFrame) or isinstance(X, pd.Series): X = np.asarray(X)
        return np.mean([m.predict(X) for m in self.models_], axis=0)

class CatBoostRegressorWrapper(MyCatBoostRegressor):
    def fit(self, X, y, verbose=False):
        if isinstance(X, torch.Tensor) or isinstance(X, pd.DataFrame) or isinstance(X, pd.Series): X = np.asarray(X)
        if isinstance(y, torch.Tensor) or isinstance(y, pd.DataFrame) or isinstance(y, pd.Series): y = np.asarray(y)
        if X.ndim > 1 and X.shape[1] > 1:
            enc = sklearn.preprocessing.OneHotEncoder(
                sparse_output=False, 
                categories=[np.arange(X.shape[1])] 
            )
            y_encoded = enc.fit_transform(y.reshape(-1, 1))
            Z = y_encoded - X
        else:
            Z = y.ravel() - X.ravel()
            if X.ndim == 2: Z = Z.reshape(-1, 1)
        return super().fit(X, Z, verbose=verbose)

    def predict_proba(self, X):
        if isinstance(X, torch.Tensor) or isinstance(X, pd.DataFrame) or isinstance(X, pd.Series): X = np.asarray(X)
        Z_pred = super().predict(X)
        if X.ndim > 1 and X.shape[1] > 1:
            return Z_pred + X
        else:
            return Z_pred + X.ravel()
        
import numpy as np
import pandas as pd
import torch
import sklearn.base
import sklearn.preprocessing
import sklearn.model_selection
import multiprocessing as mp
from lightgbm import LGBMClassifier
from scipy.special import expit, softmax

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from scipy.special import expit, softmax

import numpy as np
import pandas as pd
import multiprocessing as mp
import sklearn.base
import sklearn.preprocessing
import sklearn.model_selection
from sklearn.linear_model import LogisticRegression
from scipy.special import expit, softmax
from xgboost import XGBClassifier
import xgboost as xgb


class InitLogitXGBClassifier(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    def __init__(self, n_estimators=10, max_depth=3, learning_rate=0.01, 
                 early_stopping_rounds=20, n_jobs=1, random_state=0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.early_stopping_rounds = early_stopping_rounds
        self.n_jobs = n_jobs
        self.random_state = random_state
        print("Using the new one")

    def _fit_model(self, idxs):
        """
        Trains a single fold using the Native XGBoost API to correctly handle
        base_margins in the evaluation set.
        """
        train_idx, val_idx = idxs[0], idxs[1]
    
        dtrain = xgb.DMatrix(
            self.X_.take(train_idx, 0), 
            label=self.y_[train_idx]
        )
        dval = xgb.DMatrix(
            self.X_.take(val_idx, 0), 
            label=self.y_[val_idx]
        )

        if self.init_logits_ is not None:
            dtrain.set_base_margin(self.init_logits_.take(train_idx, 0))
            dval.set_base_margin(self.init_logits_.take(val_idx, 0))

        params = {
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "eval_metric": "logloss",
            "nthread": self.n_jobs,
            "random_state": self.random_state,
            "verbosity": 0,
            # "device": "cpu",
            # "tree_method": "hist"
        }

        if len(self.classes_) == 2:
            params["objective"] = "binary:logistic"
            params["eval_metric"] = "logloss"
        else:
            params["objective"] = "multi:softmax"
            params["num_class"] = len(self.classes_)
            params["eval_metric"] = "mlogloss"

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=[(dval, "validation")],
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=False
        )
        
        return model
    
    def _get_model_proba(self, model, X, init_score=None):
        dtest = xgb.DMatrix(X)
        raw_preds = model.predict(dtest, output_margin=True) # predict(output_margin=True) returns raw logits
        
        if init_score is not None:
            if raw_preds.ndim > 1 and init_score.ndim == 1:
                raw_preds = raw_preds + init_score[:, None]
            else:
                raw_preds = raw_preds + init_score

        if len(self.classes_) == 2:
            prob_pos = expit(raw_preds)
            return np.vstack([1 - prob_pos, prob_pos]).T
        else:
            return softmax(raw_preds, axis=1)

    def fit(self, X, y, init_logits=None, verbose=False):
        if isinstance(X, (pd.DataFrame, pd.Series)): X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)): y = y.values
        if init_logits is not None:
            if isinstance(init_logits, (pd.DataFrame, pd.Series)): 
                init_logits = init_logits.values
            init_logits = init_logits.astype(np.float32)

        self.verbose_ = verbose
        self.init_logits_ = init_logits
        
        self.le_ = sklearn.preprocessing.LabelEncoder().fit(y)
        self.X_ = X
        self.y_ = self.le_.transform(y)
        self.classes_ = self.le_.classes_
        
        splits = list(sklearn.model_selection.StratifiedKFold(
            n_splits=8, shuffle=True, random_state=self.random_state
        ).split(X, y))
        
        pool_size = min(len(splits), mp.cpu_count())
        ctx = mp.get_context('spawn') 
        with ctx.Pool(processes=pool_size) as pool:
        # with mp.Pool(processes=min(len(splits), mp.cpu_count())) as pool:
            self.models_ = pool.map(self._fit_model, splits)
        
        oof_preds_list = []
        val_indices_list = []
        for m, idxs in zip(self.models_, splits):
            val_idx = idxs[1]
            val_init = self.init_logits_.take(val_idx, 0) if self.init_logits_ is not None else None
            
            val_pred = self._get_model_proba(m, X.take(val_idx, 0), val_init)
            oof_preds_list.append(val_pred)
            val_indices_list.append(val_idx)

        oof_preds = np.concatenate(oof_preds_list, axis=0)
        val_indices = np.concatenate(val_indices_list)
        
        oof_labels = self.y_[val_indices]
        self.calib_ = get_calibrator('logistic', calibrate_with_mixture=True,
                                     logistic_binary_type='quadratic').fit(oof_preds, oof_labels)
        return self

    def predict_proba(self, X, init_logits=None):
        if isinstance(X, (pd.DataFrame, pd.Series)): X = X.values
        if init_logits is not None:
            if isinstance(init_logits, (pd.DataFrame, pd.Series)): 
                init_logits = init_logits.values
            init_logits = init_logits.astype(np.float32)

        probas_list = [self._get_model_proba(m, X, init_logits) for m in self.models_]
        avg_probas = np.mean(probas_list, axis=0)
        
        return self.calib_.predict_proba(avg_probas)

    def predict(self, X, init_logits=None):
        probas = self.predict_proba(X, init_logits=init_logits)
        return self.classes_[np.argmax(probas, axis=1)]

class InitLogitCatboostClassifier(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    def __init__(self, iterations=10, early_stopping_rounds=300, thread_count=1, random_state=0):
        self.iterations = iterations
        self.early_stopping_rounds = early_stopping_rounds
        self.thread_count = thread_count
        self.random_state = random_state

    def _fit_model(self, idxs):
        train_idx, val_idx = idxs[0], idxs[1]
        X_train, y_train = self.X_.take(train_idx, 0), self.y_[train_idx]
        X_val, y_val = self.X_.take(val_idx, 0), self.y_[val_idx]

        train_pool = Pool(X_train, label=y_train)
        val_pool = Pool(X_val, label=y_val)

        if self.init_logits_ is not None:
            train_pool.set_baseline(self.init_logits_.take(train_idx, 0))
            val_pool.set_baseline(self.init_logits_.take(val_idx, 0))

        m = CatBoostClassifier(
            iterations=self.iterations,
            random_state=self.random_state,
            early_stopping_rounds=self.early_stopping_rounds,
            thread_count=self.thread_count,
            verbose=0 if not self.verbose_ else 1
        )

        return m.fit(train_pool, eval_set=val_pool)

    def _get_model_proba(self, model, X, init_score=None):
        """Calculates probabilities by adding baseline to raw residuals."""
        raw_preds = model.predict(X, prediction_type='RawFormulaVal')
        
        # Add the initial starting point (logits)
        if init_score is not None:
            if len(self.classes_) == 2 and init_score.ndim == 2:
                init_score = init_score.ravel()
            raw_preds = raw_preds + init_score

        if len(self.classes_) == 2:
            prob_pos = expit(raw_preds)
            return np.vstack([1 - prob_pos, prob_pos]).T
        else:
            return softmax(raw_preds, axis=1)

    def fit(self, X, y, init_logits=None, verbose=False):
        if isinstance(X, (pd.DataFrame, pd.Series)): X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)): y = y.values
        if init_logits is not None and isinstance(init_logits, (pd.DataFrame, pd.Series)): 
            init_logits = init_logits.values

        self.verbose_ = verbose
        self.init_logits_ = init_logits
        self.le_ = sklearn.preprocessing.LabelEncoder().fit(y)
        self.X_, self.y_, self.classes_ = X, self.le_.transform(y), self.le_.classes_
        
        splits = list(sklearn.model_selection.StratifiedKFold(
            n_splits=8, shuffle=True, random_state=self.random_state
        ).split(X, y))
        
        with mp.Pool(processes=min(len(splits), mp.cpu_count())) as pool:
            self.models_ = pool.map(self._fit_model, splits)
        
        oof_preds_list = []
        for m, idxs in zip(self.models_, splits):
            val_idx = idxs[1]
            val_init = self.init_logits_.take(val_idx, 0) if self.init_logits_ is not None else None
            oof_preds_list.append(self._get_model_proba(m, X.take(val_idx, 0), val_init))

        oof_preds = np.concatenate(oof_preds_list, axis=0)
        val_indices = np.concatenate([idxs[1] for idxs in splits])
        oof_labels = y[val_indices]
        
        self.calib_ = get_calibrator('logistic', calibrate_with_mixture=True,
                                     logistic_binary_type='quadratic').fit(oof_preds, oof_labels)
        return self

    def predict_proba(self, X, init_logits=None):
        if isinstance(X, (pd.DataFrame, pd.Series)): X = X.values
        if init_logits is not None and isinstance(init_logits, (pd.DataFrame, pd.Series)): 
            init_logits = init_logits.values

        avg_probas = np.mean([self._get_model_proba(m, X, init_logits) for m in self.models_], axis=0)
        return self.calib_.predict_proba(avg_probas)

    def predict(self, X, init_logits=None):
        return self.le_.inverse_transform(np.argmax(self.predict_proba(X, init_logits=init_logits), axis=1))

class InitLogitLGBMClassifier(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    def _fit_model(self, idxs):
        train_idx, val_idx = idxs[0], idxs[1]
        X_train = self.X_.take(train_idx, 0)
        y_train = self.y_[train_idx]
        X_val = self.X_.take(val_idx, 0)
        y_val = self.y_[val_idx]

        fit_params = {}
        if self.init_logits_ is not None:
            fit_params['init_score'] = self.init_logits_.take(train_idx, 0)
            fit_params['eval_init_score'] = [self.init_logits_.take(val_idx, 0)]

        m = LGBMClassifier(n_estimators=10, learning_rate=0.04, subsample=0.75, subsample_freq=1, num_leaves=50,
                           random_state=0, early_stopping_round=100, min_child_samples=40, min_child_weight=1e-7,
                           n_jobs=1, verbosity=self.verbose_)
        
        return m.fit(X_train, y_train, eval_set=(X_val, y_val), **fit_params)

    def _get_model_proba(self, model, X, init_score=None):
        raw_preds = model.predict(X, raw_score=True)
        if init_score is not None:
            if raw_preds.ndim == 1:
                raw_preds = raw_preds + init_score.ravel()
            else:
                raw_preds = raw_preds + init_score
        if len(self.classes_) == 2:
            prob_pos = expit(raw_preds)
            return np.vstack([1 - prob_pos, prob_pos]).T
        else:
            return softmax(raw_preds, axis=1)

    def fit(self, X, y, init_logits=None, verbose=-1):
        if isinstance(X, torch.Tensor) or isinstance(X, pd.DataFrame) or isinstance(X, pd.Series): X = np.asarray(X)
        if isinstance(y, torch.Tensor) or isinstance(y, pd.DataFrame) or isinstance(y, pd.Series): y = np.asarray(y)
        if init_logits is not None:
            if isinstance(init_logits, torch.Tensor) or isinstance(init_logits, pd.DataFrame) or isinstance(init_logits, pd.Series): 
                init_logits = np.asarray(init_logits)

        self.verbose_ = verbose
        self.init_logits_ = init_logits
        self.le_ = sklearn.preprocessing.LabelEncoder().fit(y)
        self.X_, self.y_, self.classes_ = X, self.le_.transform(y), self.le_.classes_
        
        splits = list(sklearn.model_selection.StratifiedKFold(n_splits=8, shuffle=True, random_state=0).split(X, y))
        
        with mp.Pool(processes=min(len(splits), mp.cpu_count())) as pool:
            self.models_ = pool.map(self._fit_model, splits)
        
        oof_preds_list = []
        for m, idxs in zip(self.models_, splits):
            val_idx = idxs[1]
            val_init_score = self.init_logits_.take(val_idx, 0) if self.init_logits_ is not None else None
            oof_preds_list.append(self._get_model_proba(m, X.take(val_idx, 0), val_init_score))

        oof_preds = np.concatenate(oof_preds_list, axis=0)
        oof_labels = np.concatenate([y[idxs[1]] for idxs in splits], axis=0)
        
        self.calib_ = get_calibrator('logistic', calibrate_with_mixture=True,
                                     logistic_binary_type='quadratic').fit(oof_preds, oof_labels)
        return self

    def predict_proba(self, X, init_logits=None):
        if isinstance(X, torch.Tensor) or isinstance(X, pd.DataFrame) or isinstance(X, pd.Series): X = np.asarray(X)
        
        if init_logits is not None:
             if isinstance(init_logits, torch.Tensor) or isinstance(init_logits, pd.DataFrame) or isinstance(init_logits, pd.Series): 
                init_logits = np.asarray(init_logits)

        avg_probas = np.mean([self._get_model_proba(m, X, init_logits) for m in self.models_], axis=0)
        
        return self.calib_.predict_proba(avg_probas)

    def predict(self, X, init_logits=None):
        if isinstance(X, torch.Tensor) or isinstance(X, pd.DataFrame) or isinstance(X, pd.Series): X = np.asarray(X)
        return self.le_.inverse_transform(np.argmax(self.predict_proba(X, init_logits=init_logits), axis=1))

class PartitionWisePredictor(BaseEstimator, ClassifierMixin):
    """
    A custom meta-estimator that partitions the feature space using K-Means 
    and predicts the target variable based on the mean of the training samples 
    within that specific partition (cluster).
    
    For binary classification (Y=0 or 1), the predicted value is the proportion 
    of class 1 samples in the cluster (a probability estimate).
    """
    def __init__(self, n_clusters=3, verbose=False, **kwargs):
        """
        Initializes the PartitionWisePredictor using the Cluster Mean strategy.

        Args:
            n_clusters (int): The number of partitions (clusters) to create.
            **kwargs: Ignored, as no base estimator is used.
        """
        self.n_clusters = n_clusters
        self.clusterer = KMeans(n_clusters=self.n_clusters, random_state=42, n_init='auto')
        self.cluster_means_ = {} 
        self.verbose = verbose

    def fit(self, X, y):
        """
        Fits the clusterer and calculates the mean target value for each cluster.
        """
        if X.shape[0] < self.n_clusters:
            print("Warning: Number of samples is less than n_clusters. Using global mean as fallback.")
            self.global_mean_ = y.mean()
            self.clusterer = None # Disable clustering
            return self

        if self.verbose: 
            print(f"Fitting K-Means to partition data into {self.n_clusters} clusters...")
        cluster_labels = self.clusterer.fit_predict(X)
        
        # 2. Calculate the mean Y value for each partition
        self.cluster_means_ = {}
        for i in range(self.n_clusters):
            partition_indices = np.where(cluster_labels == i)[0]
            y_partition = y[partition_indices]

            if len(y_partition) > 0:
                mean_y = y_partition.mean()
                self.cluster_means_[i] = mean_y
                if self.verbose:
                    print(f"Partition {i} mean Y value: {mean_y:.4f} (based on {len(y_partition)} samples)")
            else:
                self.cluster_means_[i] = 0.5
                if self.verbose: 
                    print(f"Warning: Partition {i} is empty. Assigned neutral mean (0.5).")

        return self

    def predict(self, X):
        """
        Predicts the labels by first identifying the partition for each sample 
        and then using the corresponding stored mean Y value.
        
        Returns the raw probability (mean Y) for classification.
        """
        if self.clusterer is None:
            return np.full(X.shape[0], self.global_mean_)

        cluster_assignments = self.clusterer.predict(X)
        
        y_pred_proba = np.zeros(X.shape[0], dtype=np.float64)
        
        for i in range(self.n_clusters):
            test_indices_in_partition = np.where(cluster_assignments == i)[0]
            
            if len(test_indices_in_partition) > 0:
                mean_value = self.cluster_means_.get(i, 0.5) # Default to 0.5 if not found
                y_pred_proba[test_indices_in_partition] = mean_value
                
        return (y_pred_proba > 0.5).astype(int)
    
    def predict_proba(self, X):
        """
        Returns the probability of class 1 (the mean Y value).
        """
        if self.clusterer is None:
            return np.full((X.shape[0], 2), [1 - self.global_mean_, self.global_mean_])

        cluster_assignments = self.clusterer.predict(X)
        y_proba_1 = np.zeros(X.shape[0], dtype=np.float64)
        
        for i in range(self.n_clusters):
            test_indices_in_partition = np.where(cluster_assignments == i)[0]
            if len(test_indices_in_partition) > 0:
                mean_value = self.cluster_means_.get(i, 0.5)
                y_proba_1[test_indices_in_partition] = mean_value
                
        y_proba_0 = 1.0 - y_proba_1
        return np.column_stack([y_proba_0, y_proba_1])
    
class MulticlassPartitionWisePredictor(BaseEstimator, ClassifierMixin):
    """
    A custom meta-estimator that partitions the feature space using K-Means 
    and predicts the target class based on the mode or class distribution 
    within that specific partition.
    """
    def __init__(self, n_clusters=3, verbose=False, random_state=42):
        self.n_clusters = n_clusters
        self.verbose = verbose
        self.random_state = random_state
        self.clusterer = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init='auto')
        self.cluster_probs_ = {} 

    def fit(self, X, y):
        # Check that X and y have correct shape and handle multiclass labels
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        
        if X.shape[0] < self.n_clusters:
            if self.verbose: print("Warning: Small sample size. Using global distribution.")
            self.global_dist_ = np.bincount(y, minlength=self.n_classes_) / len(y)
            self.clusterer = None
            return self

        # Fit K-Means
        cluster_labels = self.clusterer.fit_predict(X)
        
        # Map each cluster to its class probability distribution
        self.cluster_probs_ = {}
        for i in range(self.n_clusters):
            partition_indices = np.where(cluster_labels == i)[0]
            y_partition = y[partition_indices]

            if len(y_partition) > 0:
                # Count occurrences of each class in this cluster
                counts = np.array([np.sum(y_partition == c) for c in self.classes_])
                probs = counts / len(y_partition)
                self.cluster_probs_[i] = probs
                
                if self.verbose:
                    top_class = self.classes_[np.argmax(probs)]
                    print(f"Cluster {i}: Top Class {top_class} (p={np.max(probs):.2f})")
            else:
                # Uniform distribution fallback for empty clusters
                self.cluster_probs_[i] = np.ones(self.n_classes_) / self.n_classes_

        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        
        if self.clusterer is None:
            return np.tile(self.global_dist_, (X.shape[0], 1))

        # Identify clusters for test data
        cluster_assignments = self.clusterer.predict(X)
        y_proba = np.zeros((X.shape[0], self.n_classes_))
        
        for i in range(self.n_clusters):
            indices = np.where(cluster_assignments == i)[0]
            if len(indices) > 0:
                # Assign the pre-calculated probability vector for this cluster
                y_proba[indices] = self.cluster_probs_.get(i, np.ones(self.n_classes_) / self.n_classes_)
                
        return y_proba

    def predict(self, X):
        # Get probabilities and return the class with the highest probability
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
    
class IsotonicPredictor:
    def __init__(self):
        self.cal = get_calibrator('isotonic')

    def fit(self, p, y):
      self.cal.fit(p, y)

    def predict_proba(self, p):
        return self.cal.predict_proba(p)
    
class IsotonicPredictorBinary:
    def __init__(self):
        self.cal = get_calibrator('isotonic')

    def fit(self, p, y):
        p_1d = p.flatten()
        p_2d = np.stack([1 - p_1d, p_1d], axis=1)
        self.cal.fit(p_2d, y)

    def predict_proba(self, p):
        p_1d = p.flatten()
        p_2d = np.stack([1 - p_1d, p_1d], axis=1)
        return self.cal.predict_proba(p_2d)
  
class TSPredictorBinary:
    def __init__(self):
        self.cal = get_calibrator('temp-scaling')

    def fit(self, p, y):
        p_1d = p.flatten()
        p_2d = np.stack([1 - p_1d, p_1d], axis=1)
        self.cal.fit(p_2d, y)

    def predict_proba(self, p):
        p_1d = p.flatten()
        p_2d = np.stack([1 - p_1d, p_1d], axis=1)
        return self.cal.predict_proba(p_2d)
   
class SMSPredictor:
    def __init__(self):
        self.cal = get_calibrator('logistic')

    def fit(self, p, y):
      self.cal.fit(p, y)

    def predict_proba(self, p):
        return self.cal.predict_proba(p)
    
class TSPredictor:
    def __init__(self):
        self.cal = get_calibrator('temp-scaling')

    def fit(self, p, y):
      self.cal.fit(p, y)

    def predict_proba(self, p):
        return self.cal.predict_proba(p)


class WSCatboostClassifier(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    def __init__(self, iterations=10, use_init_logits=True, early_stopping_rounds=300, thread_count=1, random_state=0):
        self.iterations = iterations
        self.early_stopping_rounds = early_stopping_rounds
        self.thread_count = thread_count
        self.random_state = random_state
        self.use_init_logits = use_init_logits

    def _fit_model(self, idxs):
        train_idx, val_idx = idxs[0], idxs[1]
        X_train, y_train = self.X_.take(train_idx, 0), self.y_[train_idx]
        X_val, y_val = self.X_.take(val_idx, 0), self.y_[val_idx]

        train_pool = Pool(X_train, label=y_train)
        val_pool = Pool(X_val, label=y_val)

        if self.init_logits_ is not None:
            train_pool.set_baseline(self.init_logits_.take(train_idx, 0))
            val_pool.set_baseline(self.init_logits_.take(val_idx, 0))

        m = CatBoostClassifier(
            iterations=self.iterations,
            random_state=self.random_state,
            early_stopping_rounds=self.early_stopping_rounds,
            thread_count=self.thread_count,
            verbose=0 if not self.verbose_ else 1
        )

        return m.fit(train_pool, eval_set=val_pool)

    def _get_model_proba(self, model, X):
        """Calculates probabilities by adding baseline to raw residuals."""
        raw_preds = model.predict(X, prediction_type='RawFormulaVal')
        if self.use_init_logits:
            if X.shape[-1] == 1:
                p_1d = X.flatten()
                p_2d = np.stack([1 - p_1d, p_1d], axis=1)
                X = self.init_cal.predict_proba(p_2d)[:, [1]]
                init_score = probs_to_logits(X)
            else:
                X = self.init_cal.predict_proba(X)
                init_score = probs_to_logits(X)
                if X.shape[1] == 2:
                    init_score = init_score[:, 1]
        else:
            init_score = None
        
        # Add the initial starting point (logits)
        if init_score is not None:
            if len(self.classes_) == 2 and init_score.ndim == 2:
                init_score = init_score.ravel()
            raw_preds = raw_preds + init_score

        if len(self.classes_) == 2:
            prob_pos = expit(raw_preds)
            return np.vstack([1 - prob_pos, prob_pos]).T
        else:
            return softmax(raw_preds, axis=1)

    def fit(self, X, y, verbose=False):
        if isinstance(X, (pd.DataFrame, pd.Series)): X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)): y = y.values
        
        if self.use_init_logits:
            if X.shape[-1] == 1:
                self.init_cal = get_calibrator('temp-scaling')
                p_1d = X.flatten()
                p_2d = np.stack([1 - p_1d, p_1d], axis=1)
                self.init_cal.fit(p_2d, y)
                X_init = self.init_cal.predict_proba(p_2d)[:, [1]]
                init_logits = probs_to_logits(X_init)
            else:
                self.init_cal = get_calibrator('temp-scaling')
                self.init_cal.fit(X, y)
                X_init = self.init_cal.predict_proba(X)
                init_logits = probs_to_logits(X_init)
                if X.shape[1] == 2:
                    init_logits = init_logits[:, 1]
        else:
            init_logits = None

        self.verbose_ = verbose
        self.init_logits_ = init_logits
        self.le_ = sklearn.preprocessing.LabelEncoder().fit(y)
        self.X_, self.y_, self.classes_ = X, self.le_.transform(y), self.le_.classes_
        
        splits = list(sklearn.model_selection.StratifiedKFold(
            n_splits=8, shuffle=True, random_state=self.random_state
        ).split(X, y))
        
        with mp.Pool(processes=min(len(splits), mp.cpu_count())) as pool:
            self.models_ = pool.map(self._fit_model, splits)
        
        oof_preds_list = []
        for m, idxs in zip(self.models_, splits):
            val_idx = idxs[1]
            oof_preds_list.append(self._get_model_proba(m, X.take(val_idx, 0)))

        oof_preds = np.concatenate(oof_preds_list, axis=0)
        val_indices = np.concatenate([idxs[1] for idxs in splits])
        oof_labels = y[val_indices]
        
        self.calib_ = get_calibrator('logistic', calibrate_with_mixture=True,
                                     logistic_binary_type='quadratic').fit(oof_preds, oof_labels)
        return self

    def predict_proba(self, X):
        if isinstance(X, (pd.DataFrame, pd.Series)): X = X.values
        avg_probas = np.mean([self._get_model_proba(m, X) for m in self.models_], axis=0)
        return self.calib_.predict_proba(avg_probas)

    def predict(self, X):
        return self.le_.inverse_transform(np.argmax(self.predict_proba(X), axis=1))

class WSLGBMClassifier(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    def __init__(self, n_estimators=10, use_init_logits=True, learning_rate=0.04, subsample=0.75, num_leaves=50, subsample_freq=1, random_state=0, early_stopping_round=100, min_child_samples=40, min_child_weight=1e-7, n_jobs=1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.num_leaves = num_leaves
        self.subsample_freq = subsample_freq
        self.random_state = random_state
        self.early_stopping_round = early_stopping_round
        self.min_child_samples = min_child_samples
        self.min_child_weight = min_child_weight
        self.n_jobs = n_jobs
        self.use_init_logits = use_init_logits


    def _fit_model(self, idxs):
        train_idx, val_idx = idxs[0], idxs[1]
        X_train = self.X_.take(train_idx, 0)
        y_train = self.y_[train_idx]
        X_val = self.X_.take(val_idx, 0)
        y_val = self.y_[val_idx]

        fit_params = {}
        if self.init_logits_ is not None:
            fit_params['init_score'] = self.init_logits_.take(train_idx, 0)
            fit_params['eval_init_score'] = [self.init_logits_.take(val_idx, 0)]

        m = LGBMClassifier(n_estimators=self.n_estimators, learning_rate=self.learning_rate, subsample=self.subsample, subsample_freq=self.subsample_freq, num_leaves=self.num_leaves,
                           random_state=self.random_state, early_stopping_round=self.early_stopping_round, min_child_samples=self.min_child_samples, min_child_weight=self.min_child_weight,
                           n_jobs=self.n_jobs, verbosity=self.verbose_)
        
        return m.fit(X_train, y_train, eval_set=(X_val, y_val), **fit_params)

    def _get_model_proba(self, model, X):
        if self.use_init_logits:
            if X.shape[-1] == 1:
                p_1d = X.flatten()
                p_2d = np.stack([1 - p_1d, p_1d], axis=1)
                X = self.init_cal.predict_proba(p_2d)[:, [1]]
                init_score = probs_to_logits(X)
            else:
                X = self.init_cal.predict_proba(X)
                init_score = probs_to_logits(X)
                if X.shape[1] == 2:
                    init_score = init_score[:, 1]

        raw_preds = model.predict(X, raw_score=True)
        if init_score is not None:
            if raw_preds.ndim == 1:
                raw_preds = raw_preds + init_score.ravel()
            else:
                raw_preds = raw_preds + init_score
        if len(self.classes_) == 2:
            prob_pos = expit(raw_preds)
            return np.vstack([1 - prob_pos, prob_pos]).T
        else:
            return softmax(raw_preds, axis=1)

    def fit(self, X, y, verbose=-1):
        if isinstance(X, torch.Tensor) or isinstance(X, pd.DataFrame) or isinstance(X, pd.Series): X = np.asarray(X)
        if isinstance(y, torch.Tensor) or isinstance(y, pd.DataFrame) or isinstance(y, pd.Series): y = np.asarray(y)
        
        if self.use_init_logits:
            if X.shape[-1] == 1:
                self.init_cal = get_calibrator('temp-scaling')
                p_1d = X.flatten()
                p_2d = np.stack([1 - p_1d, p_1d], axis=1)
                self.init_cal.fit(p_2d, y)
                X_init = self.init_cal.predict_proba(p_2d)[:, [1]]
                init_logits = probs_to_logits(X_init)
            else:
                self.init_cal = get_calibrator('temp-scaling')
                self.init_cal.fit(X, y)
                X_init = self.init_cal.predict_proba(X)
                init_logits = probs_to_logits(X_init)
                if X.shape[1] == 2:
                    init_logits = init_logits[:, 1]

        self.verbose_ = verbose
        self.init_logits_ = init_logits
        self.le_ = sklearn.preprocessing.LabelEncoder().fit(y)
        self.X_, self.y_, self.classes_ = X, self.le_.transform(y), self.le_.classes_
        
        splits = list(sklearn.model_selection.StratifiedKFold(n_splits=8, shuffle=True, random_state=0).split(X, y))
        
        with mp.Pool(processes=min(len(splits), mp.cpu_count())) as pool:
            self.models_ = pool.map(self._fit_model, splits)
        
        oof_preds_list = []
        for m, idxs in zip(self.models_, splits):
            val_idx = idxs[1]
            val_init_score = self.init_logits_.take(val_idx, 0) if self.init_logits_ is not None else None
            oof_preds_list.append(self._get_model_proba(m, X.take(val_idx, 0)))

        oof_preds = np.concatenate(oof_preds_list, axis=0)
        oof_labels = np.concatenate([y[idxs[1]] for idxs in splits], axis=0)
        
        self.calib_ = get_calibrator('logistic', calibrate_with_mixture=True,
                                     logistic_binary_type='quadratic').fit(oof_preds, oof_labels)
        return self

    def predict_proba(self, X):
        if isinstance(X, torch.Tensor) or isinstance(X, pd.DataFrame) or isinstance(X, pd.Series): X = np.asarray(X)
        
        avg_probas = np.mean([self._get_model_proba(m, X) for m in self.models_], axis=0)
        
        return self.calib_.predict_proba(avg_probas)

    def predict(self, X):
        if isinstance(X, torch.Tensor) or isinstance(X, pd.DataFrame) or isinstance(X, pd.Series): X = np.asarray(X)
        return self.le_.inverse_transform(np.argmax(self.predict_proba(X), axis=1))





# def insert_missing_class_columns(y_pred: torch.Tensor, train_ds: DictDataset) -> torch.Tensor:
#     """
#     If train_ds.tensors['y'] does not contain some of the classes specified in train_ds.tensor_infos['y']
#     and if y_pred does not contain columns for these missing classes,
#     add columns for the missing classes to y_pred, with small probabilities.
#     :param y_pred: Tensor of logits, shape [n_batch, n_classes]
#     :param train_ds: Dataset used for training the model that produced y_pred.
#     :return: Returns y_pred with possibly some columns inserted.
#     """
#     n_classes = train_ds.tensor_infos['y'].get_cat_sizes()[0].item()
#     if y_pred.shape[-1] >= n_classes:
#         return y_pred  # already all columns

#     # assume that the missing classes/columns in y_pred are exactly those that are not represented in the training set
#     train_class_counts = torch.bincount(train_ds.tensors['y'].squeeze(-1), minlength=n_classes).cpu()
#     n_missing = n_classes - y_pred.shape[-1]
#     pred_col_idx = 0
#     new_cols = []
#     logsumexp = torch.logsumexp(y_pred, dim=-1)
#     # expected posterior probability of the class under uniform prior
#     # (expected value of corresponding Dirichlet distribution, which is conjugate prior to "multinoulli" distribution)
#     posterior_prob = 1 / (train_ds.n_samples + n_classes)
#     # ensure that the probability of missing classes is posterior_prob if y_pred are the logits
#     missing_values = logsumexp + np.log(posterior_prob / (1 - posterior_prob * n_missing))
#     for i in range(n_classes):
#         if train_class_counts[i] > 0:
#             # this column should be represented
#             new_cols.append(y_pred[:, pred_col_idx])
#             pred_col_idx += 1
#         else:
#             new_cols.append(missing_values)

#     return torch.stack(new_cols, dim=-1)

def probs_to_logits(probs: np.ndarray) -> np.ndarray:
    """Converts multiclass probabilities to logits using the log function.
    Clips logits to the log of the tinyest normal float32 to avoid infinite logit values.
    """
    probs = probs.astype(np.float64)
    thresh = np.log(np.finfo(np.float64).tiny)
    with np.errstate(divide="ignore"):
        logits = np.log(probs)
    logits = np.clip(logits, a_min=thresh, a_max=None)
    return logits