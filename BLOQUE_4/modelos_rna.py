"""Módulo de inferencia y deserialización portable para modelos RNA y Baselines de Mantos Cobrizo.
Bloque 4 - Metodología v4.0 (Universidad de Antofagasta).
"""

import numpy as np
import joblib
from sklearn.base import BaseEstimator, RegressorMixin

def predecir_rna_mlp2(artefacto_o_path, X):
    """Genera predicciones con el ensamble multi-semilla de RNA_MLP_2.
    
    Parameters
    ----------
    artefacto_o_path : dict or str
        Diccionario serializado o ruta al archivo .joblib.
    X : np.ndarray de forma (n_samples, 2)
        Matriz con columnas [Cancha_CuT, Cancha_CuS].
        
    Returns
    -------
    np.ndarray de forma (n_samples,)
        Predicciones de Canje_CuT en %Cu.
    """
    if isinstance(artefacto_o_path, str):
        artefacto = joblib.load(artefacto_o_path)
    else:
        artefacto = artefacto_o_path
        
    scaler = artefacto['scaler']
    estimadores = artefacto['estimadores']
    
    X_scaled = scaler.transform(X)
    preds = np.column_stack([m.predict(X_scaled) for m in estimadores])
    return np.mean(preds, axis=1)

def predecir_residual_mlp(artefacto_o_path, X):
    """Genera predicciones con el ensamble residual (Skip-Connection) de Res_MLP_2_2.
    
    Parameters
    ----------
    artefacto_o_path : dict or str
        Diccionario serializado o ruta al archivo .joblib.
    X : np.ndarray de forma (n_samples, 2)
        Matriz con columnas [Cancha_CuT, Cancha_CuS].
        
    Returns
    -------
    np.ndarray de forma (n_samples,)
        Predicciones de Canje_CuT en %Cu calculadas como X[:, 0] - e_hat.
    """
    if isinstance(artefacto_o_path, str):
        artefacto = joblib.load(artefacto_o_path)
    else:
        artefacto = artefacto_o_path
        
    scaler = artefacto['scaler']
    estimadores = artefacto['estimadores']
    
    X_scaled = scaler.transform(X)
    bias_preds = np.column_stack([m.predict(X_scaled) for m in estimadores])
    mean_bias = np.mean(bias_preds, axis=1)
    return X[:, 0] - mean_bias

def predecir_baseline_1b(artefacto_o_path, X):
    """Genera predicciones con la corrección de sesgo constante (Baseline 1B).
    
    Parameters
    ----------
    artefacto_o_path : dict or str
        Diccionario serializado o ruta al archivo .joblib.
    X : np.ndarray de forma (n_samples, >=1)
        Matriz con la primera columna correspondiente a Cancha_CuT.
        
    Returns
    -------
    np.ndarray de forma (n_samples,)
        Predicciones de Canje_CuT = Cancha_CuT - Sesgo_Constante.
    """
    if isinstance(artefacto_o_path, str):
        artefacto = joblib.load(artefacto_o_path)
    else:
        artefacto = artefacto_o_path
        
    sesgo = artefacto['Sesgo_Constante']
    cancha_cut = X[:, 0] if X.ndim > 1 else X
    return cancha_cut - sesgo

class PredictorWrapper(BaseEstimator, RegressorMixin):
    """Envoltorio compatible con scikit-learn (fit, predict, score) para SHAP y Permutación."""
    def __init__(self, artefacto=None, es_residual=False):
        self.artefacto = artefacto
        self.es_residual = es_residual
        
    def fit(self, X, y=None):
        return self
        
    def predict(self, X):
        if self.es_residual:
            return predecir_residual_mlp(self.artefacto, X)
        else:
            return predecir_rna_mlp2(self.artefacto, X)
