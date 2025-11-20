"""
Módulo de regresión lineal
"""
import numpy as np
from typing import Tuple, Optional


class LinearRegression:
    """Implementación de regresión lineal simple"""
    
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.X_train = None
        self.y_train = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Entrena el modelo de regresión lineal
        
        Args:
            X: Variable independiente (features)
            y: Variable dependiente (target)
            
        Returns:
            Self para method chaining
        """
        # Asegurar que X sea 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.X_train = X
        self.y_train = y
        
        # Calcular coeficientes usando mínimos cuadrados
        # y = mx + b
        # m = (n*Σ(xy) - Σx*Σy) / (n*Σ(x²) - (Σx)²)
        # b = (Σy - m*Σx) / n
        
        n = len(X)
        x = X.flatten()
        
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x ** 2)
        
        # Calcular pendiente (coeficiente)
        self.coef_ = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        
        # Calcular intercepto
        self.intercept_ = (sum_y - self.coef_ * sum_x) / n
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones usando el modelo entrenado
        
        Args:
            X: Variable independiente para predicción
            
        Returns:
            Array con predicciones
        """
        if self.coef_ is None or self.intercept_ is None:
            raise ValueError("El modelo debe ser entrenado primero usando fit()")
        
        # Asegurar que X sea 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        return self.coef_ * X.flatten() + self.intercept_
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calcula el coeficiente de determinación R² del modelo
        
        Args:
            X: Variable independiente
            y: Variable dependiente real
            
        Returns:
            Valor R² (0-1, donde 1 es perfecto)
        """
        y_pred = self.predict(X)
        
        # R² = 1 - (SS_res / SS_tot)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        return 1 - (ss_res / ss_tot)
    
    def get_equation(self) -> str:
        """
        Obtiene la ecuación de la línea de regresión
        
        Returns:
            String con la ecuación en formato y = mx + b
        """
        if self.coef_ is None or self.intercept_ is None:
            return "Modelo no entrenado"
        
        sign = "+" if self.intercept_ >= 0 else ""
        return f"y = {self.coef_:.4f}x {sign}{self.intercept_:.4f}"
