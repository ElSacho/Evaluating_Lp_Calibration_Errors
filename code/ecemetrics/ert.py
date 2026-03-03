from sklearn.model_selection import StratifiedKFold, KFold
import torch
import numpy as np
import pandas as pd
import inspect
import warnings

from ecemetrics.check import *
from ecemetrics.classifiers import CheapLGBMClassifier
from ecemetrics.losses import *

class ERT:
    def __init__(self, model_cls=CheapLGBMClassifier, **model_kwargs):    
        """
        Initialize the Excess Risk of the Target yage metric. 

        model_cls: the class of the model (e.g., RandomForestClassifier, CatBoostClassifier)
        model_kwargs: keyword arguments to initialize the model
        """
        self.model_cls = model_cls
        self.model_kwargs = model_kwargs
        self.model = self.init_model()
        self.fitted = False
        self.added_losses = None
        self.tab_losses = []

    def init_model(self):
        """Re-initialize the model."""
        self.model = self.model_cls(**self.model_kwargs)
        return self.model
    
    def fit(self, x_train, y_train, x_val=None, y_val=None, **fit_kwargs):
        """
        Fit the classifier
        
        :param x_train: data used to train the model (either numpy, torch or dataframe) of shape (n, d)
        :param y_train: y vector of shape (n, ) with classe indices
        :param x_val: (optional) additional data used to train the model (either numpy, torch or dataframe) of shape (n, d)
        :param y_val:(optional) additional y vector used to train the model (either numpy, torch or dataframe) of shape (n,)
        :param fit_kwargs: (optional) arguments that the model needs to use to fit the classifier
        """
        check_tabular(x_train)
        check_y(y_train)
        check_consistency(y_train, x_train)
        if x_val is not None:
            check_tabular(x_val)
            check_y(y_val)
            check_consistency(y_val, x_val)

        if y_val is not None:
            self.model.fit(x_train, y_train, X_val=x_val, y_val=y_val, **fit_kwargs)
        else:
            self.model.fit(x_train, y_train, **fit_kwargs)
       
        self.fitted = True

    def get_conditional_prediction(self, f_x, init_logits=None):
        """
        Get predicted conditional probabilities P(Y | pred )

        Returns:
        - binary classification: shape (n, 1)
        - multi-class classification: shape (n, d)
        """

        # --- Prediction ---
        if hasattr(self.model, "predict_proba"):
            if init_logits is not None:
                g_f_x = self.model.predict_proba(f_x, init_logits=init_logits)
            else:
                g_f_x = self.model.predict_proba(f_x)

            # if binary classification, discard the prediction
            if g_f_x.ndim == 2 and g_f_x.shape[1] == 2 and f_x.shape[1] == 1:
                output = g_f_x[:, 1:2]  # shape (n, 1)
            else:
                # multi-class: (n, d)
                output = g_f_x

        else:
            raise ValueError("The model should have a predict_proba function")

        # --- Type alignment with x ---
        if isinstance(f_x, pd.DataFrame):
            output = pd.DataFrame(output, index=f_x.index)

        elif isinstance(f_x, torch.Tensor):
            output = torch.tensor(output, dtype=f_x.dtype)

        elif isinstance(f_x, np.ndarray):
            output = np.asarray(output, dtype=f_x.dtype)

        return output
 
    def add_loss(self, loss):
        """
        Add a loss to the table of all proper losses you want to evaluate conditional misyage
        
        :param loss: loss function of type loss(pred, y) and returns the loss value
        """
        if self.added_losses is None:
            self.added_losses = [loss]
        else:
            self.added_losses.append(loss)

    def make_default_multiclass_losses(self):
        """
        Generate the losses you want to evaluate the ERT
        """
        return [
            make_generalized_norm_score(z=2),
            L1_ECE,
            brier_score,
            logloss
                ]
    
    def make_default_binary_losses(self):
        """
        Generate the losses you want to evaluate the ERT
        """
        return [
            L1_ECE,
            brier_score,
            logloss
                ]
       
  
    def evaluate_multiple_losses(self, preds_init, y, n_splits = 5, init_logits=None, random_state=42, tab_losses=None, **fit_kwargs):
        """
        Evaluate the loss-ERT. 
        
        :param preds: Feature vector. Either numpy, torch or dataframe, of shape (n, d) -> represents the predictions
        :param y: y vector with 1 and 0, where 1=(Y in C(X)). Either numpy, torch or dataframe, of shape (n,)
        :param n_splits: (optional) Default=5, Number of splits to be done. If n_splits==0 then the model as to be already learned. Otherwise n_splits needs to be integer larger (or equal) than 2.
        :param random_state: (optional) Default=42. Random seed to get reproducable results. 
        :param loss: (optional) Default=L1_ECE. loss function of type loss(pred, y) and returns the loss value 
        :param fit_kwargs: (optional) arguments that the model needs to use to fit the classifier
        
        Returns 
            Float : ERT estimated value for the loss
        """
        
        check_tabular(preds_init)
        check_y(y)
        check_consistency(y, preds_init)
        check_preds_tab_ok(preds_init, y)
        
        if tab_losses is None:
            if preds_init.shape[1] == 1:
                tab_losses = self.make_default_binary_losses()
            else:
                tab_losses = self.make_default_multiclass_losses()

        ERT_values = {"ERT_"+loss.__name__: [] for loss in tab_losses}
        
        if n_splits >= 2:
            check_n_splits(n_splits)

            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            for train_index, test_index in kf.split(preds_init):
                if isinstance(preds_init, pd.DataFrame):
                    preds_init_train, preds_init_test = preds_init.iloc[train_index], preds_init.iloc[test_index]
                else:
                    preds_init_train, preds_init_test = preds_init[train_index], preds_init[test_index]
                if isinstance(y, (pd.Series, pd.DataFrame)):
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                else:
                    y_train, y_test = y[train_index], y[test_index]
                if init_logits is not None:
                    if isinstance(init_logits, pd.DataFrame):
                        init_logits_train, init_logits_test = init_logits.iloc[train_index], init_logits.iloc[test_index]
                    else:
                        init_logits_train, init_logits_test = init_logits[train_index], init_logits[test_index]

                self.init_model()
                if init_logits is not None:
                    self.model.fit(preds_init_train, y_train, init_logits=init_logits_train, **fit_kwargs)
                else:
                    self.model.fit(preds_init_train, y_train, **fit_kwargs)
                
                if init_logits is not None:
                    preds_rectified_test = self.get_conditional_prediction(preds_init_test, init_logits=init_logits_test)
                else:
                    preds_rectified_test = self.get_conditional_prediction(preds_init_test)

                for loss in tab_losses:
                    ERT_loss = evaluate_with_predictions(preds_rectified_test, y_test, preds_init_test, loss=loss)
                    ERT_values["ERT_"+loss.__name__].append(ERT_loss)

            results = {key: np.mean(values) for key, values in ERT_values.items()}
            return results
            
        else:
            if not self.fitted:
                raise Exception("You need to first fit the model. You can evaluate with cross validation using n_splits > 1")
    
        if init_logits is not None:
            preds_rectified_test = self.get_conditional_prediction(preds_init, init_logits=init_logits)
        else:
            preds_rectified_test = self.get_conditional_prediction(preds_init)
        
        for loss in tab_losses:
            ERT_loss = evaluate_with_predictions(preds_rectified_test, y, preds_init, loss=loss)
            ERT_values["ERT_"+loss.__name__] = ERT_loss

        return ERT_values


    def evaluate_multiple_losses_old(self, preds_init, y, n_splits = 5, random_state=42, all_losses=None, **fit_kwargs):
        """
        Returns the ERT values for all losses in self.tab_losses
            
        :param x: Feature vector. Either numpy, torch or dataframe, of shape (n, d)
        :param y: y vector with 1 and 0, where 1=(Y in C(X)). Either numpy, torch or dataframe, of shape (n,)
        :param alpha: Float in (0,1). Target yage level. 
        :param n_splits: (optional) Default=5, Number of splits to be done. If n_splits==0 then the model as to be already learned. Otherwise n_splits needs to be integer larger (or equal) than 2.
        :param random_state: Integer (optional) Default=42. Random seed to get reproducable results. 
        :param all_losses: List (optional) All losses to evaluate the metrics. 
        :param fit_kwargs: (optional) arguments that the model needs to use to fit the classifier
    
        """
        check_tabular(preds_init)
        check_y(y)
        check_consistency(y, preds_init)
        check_preds_tab_ok(preds_init, y)
        
        if all_losses is None:
            self.make_losses()
            all_losses = self.tab_losses

        ERT_values = {"ERT_"+loss.__name__: [] for loss in all_losses}
        
        if n_splits >= 2:
            check_n_splits(n_splits)
            # kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            # for train_index, test_index in kf.split(preds_init, y):
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            for train_index, test_index in kf.split(preds_init):
                if isinstance(preds_init, pd.DataFrame):
                    preds_init_train, preds_init_test = preds_init.iloc[train_index], preds_init.iloc[test_index]
                else:
                    preds_init_train, preds_init_test = preds_init[train_index], preds_init[test_index]
                if isinstance(y, (pd.Series, pd.DataFrame)):
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                else:
                    y_train, y_test = y[train_index], y[test_index]

                self.init_model()
                self.model.fit(preds_init_train, y_train, **fit_kwargs)

                preds_rectified_test = self.get_conditional_prediction(preds_init_test)

                for loss in all_losses:
                    ERT_loss = evaluate_with_predictions(preds_rectified_test, y_test, preds_init_test, loss=loss)
                    ERT_values["ERT_"+loss.__name__].append(ERT_loss)
                    
            results = {key: np.mean(values) for key, values in ERT_values.items()}

            return results
        else:
            if not self.fitted:
                raise Exception("You need to first fit the model. You can evaluate with cross validation using n_splits > 1")
        
        preds_rectified = self.get_conditional_prediction(preds_init)
        
        for loss in all_losses:
            ERT_loss = evaluate_with_predictions(preds_rectified, y, preds_init, loss=loss)
            ERT_values["ERT_"+loss.__name__] = ERT_loss

        return ERT_values
  
    def evaluate(self, preds_init, y, n_splits = 5, init_logits=None, random_state=42, loss=L1_ECE, **fit_kwargs):
        """
        Evaluate the loss-ERT. 
        
        :param preds: Feature vector. Either numpy, torch or dataframe, of shape (n, d) -> represents the predictions
        :param y: y vector with 1 and 0, where 1=(Y in C(X)). Either numpy, torch or dataframe, of shape (n,)
        :param n_splits: (optional) Default=5, Number of splits to be done. If n_splits==0 then the model as to be already learned. Otherwise n_splits needs to be integer larger (or equal) than 2.
        :param random_state: (optional) Default=42. Random seed to get reproducable results. 
        :param loss: (optional) Default=L1_ECE. loss function of type loss(pred, y) and returns the loss value 
        :param fit_kwargs: (optional) arguments that the model needs to use to fit the classifier
        
        Returns 
            Float : ERT estimated value for the loss
        """
        
        check_tabular(preds_init)
        check_y(y)
        check_consistency(y, preds_init)
        check_preds_tab_ok(preds_init, y)
        # TODO: check les logits
        
        if n_splits >= 2:
            check_n_splits(n_splits)

            ERT_values = []
            # kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            # for train_index, test_index in kf.split(preds_init, y):
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            for train_index, test_index in kf.split(preds_init):
                if isinstance(preds_init, pd.DataFrame):
                    preds_init_train, preds_init_test = preds_init.iloc[train_index], preds_init.iloc[test_index]
                else:
                    preds_init_train, preds_init_test = preds_init[train_index], preds_init[test_index]
                if isinstance(y, (pd.Series, pd.DataFrame)):
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                else:
                    y_train, y_test = y[train_index], y[test_index]
                if init_logits is not None:
                    if isinstance(init_logits, pd.DataFrame):
                        init_logits_train, init_logits_test = init_logits.iloc[train_index], init_logits.iloc[test_index]
                    else:
                        init_logits_train, init_logits_test = init_logits[train_index], init_logits[test_index]

                self.init_model()
                if init_logits is not None:
                    self.model.fit(preds_init_train, y_train, init_logits=init_logits_train, **fit_kwargs)
                else:
                    self.model.fit(preds_init_train, y_train, **fit_kwargs)
                
                if init_logits is not None:
                    preds_rectified_test = self.get_conditional_prediction(preds_init_test, init_logits=init_logits_test)
                else:
                    preds_rectified_test = self.get_conditional_prediction(preds_init_test)

                ERT_loss = evaluate_with_predictions(preds_rectified_test, y_test, preds_init_test, loss=loss)
                ERT_values.append(ERT_loss)

            ERT_ell = np.mean(ERT_values)
            return float(ERT_ell)
            
        else:
            if not self.fitted:
                raise Exception("You need to first fit the model. You can evaluate with cross validation using n_splits > 1")

        if init_logits is not None:
            preds_rectified = self.get_conditional_prediction(preds_init, init_logits=init_logits)
        else:
            preds_rectified = self.get_conditional_prediction(preds_init)

        ERT_loss = evaluate_with_predictions(preds_rectified, y, preds_init, loss=loss)
        
        if isinstance(ERT_loss, torch.Tensor):
            ERT_loss = ERT_loss.item()
        return float(ERT_loss)

def evaluate_with_predictions(pred_rectified, y, preds_initial, loss=brier_score):
    """
    Docstring pour evaluate_with_predictions
    
    :param pred_rectified: Prediction rectified
    :param y: Vector with 1 and zeros (same type and lenght as x)
    :param preds_initial: initial predictions
    :param loss: Loss to be used to evaluate the metric

    Returns : 
        The risk difference between the constant predictor equal to 1-alpha and the prediction.
    """
    sig = inspect.signature(loss)
    if "f_x" in sig.parameters:
        loss_pred = loss(pred_rectified, y, f_x=preds_initial)
        loss_init = loss(preds_initial, y, f_x=preds_initial)
    else:
        loss_pred = loss(pred_rectified, y)
        loss_init = loss(preds_initial, y)

    return np.mean(np.asarray(loss_init, dtype=float)) - np.mean(np.asarray(loss_pred, dtype=float))
