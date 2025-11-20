"""
Módulo de análisis estadístico
"""
import numpy as np
from typing import Dict


class DataAnalyzer:
    """Clase para realizar análisis estadísticos de datos y regresión"""
    
    def __init__(self):
        pass
    
    def calculate_statistics(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Calcula estadísticas descriptivas de los datos
        
        Args:
            X: Variable independiente
            y: Variable dependiente
            
        Returns:
            Diccionario con estadísticas
        """
        X_flat = X.flatten()
        
        stats = {
            'X': {
                'media': float(np.mean(X_flat)),
                'mediana': float(np.median(X_flat)),
                'desviacion_std': float(np.std(X_flat)),
                'varianza': float(np.var(X_flat)),
                'minimo': float(np.min(X_flat)),
                'maximo': float(np.max(X_flat)),
                'rango': float(np.max(X_flat) - np.min(X_flat)),
                'cuartil_25': float(np.percentile(X_flat, 25)),
                'cuartil_75': float(np.percentile(X_flat, 75))
            },
            'Y': {
                'media': float(np.mean(y)),
                'mediana': float(np.median(y)),
                'desviacion_std': float(np.std(y)),
                'varianza': float(np.var(y)),
                'minimo': float(np.min(y)),
                'maximo': float(np.max(y)),
                'rango': float(np.max(y) - np.min(y)),
                'cuartil_25': float(np.percentile(y, 25)),
                'cuartil_75': float(np.percentile(y, 75))
            }
        }
        
        return stats
    
    def calculate_correlation(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calcula el coeficiente de correlación de Pearson
        
        Args:
            X: Variable independiente
            y: Variable dependiente
            
        Returns:
            Coeficiente de correlación (-1 a 1)
        """
        X_flat = X.flatten()
        correlation_matrix = np.corrcoef(X_flat, y)
        return float(correlation_matrix[0, 1])
    
    def calculate_mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcula el Error Cuadrático Medio (MSE)
        
        Args:
            y_true: Valores reales
            y_pred: Valores predichos
            
        Returns:
            MSE
        """
        return float(np.mean((y_true - y_pred) ** 2))
    
    def calculate_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcula la Raíz del Error Cuadrático Medio (RMSE)
        
        Args:
            y_true: Valores reales
            y_pred: Valores predichos
            
        Returns:
            RMSE
        """
        return float(np.sqrt(self.calculate_mse(y_true, y_pred)))
    
    def calculate_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcula el Error Absoluto Medio (MAE)
        
        Args:
            y_true: Valores reales
            y_pred: Valores predichos
            
        Returns:
            MAE
        """
        return float(np.mean(np.abs(y_true - y_pred)))
    
    def regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          r2_score: float) -> Dict:
        """
        Calcula todas las métricas de regresión
        
        Args:
            y_true: Valores reales
            y_pred: Valores predichos
            r2_score: Coeficiente R²
            
        Returns:
            Diccionario con todas las métricas
        """
        return {
            'R2': r2_score,
            'MSE': self.calculate_mse(y_true, y_pred),
            'RMSE': self.calculate_rmse(y_true, y_pred),
            'MAE': self.calculate_mae(y_true, y_pred)
        }
    
    def print_analysis(self, stats: Dict, metrics: Dict = None, 
                      correlation: float = None):
        """
        Imprime un análisis detallado de los datos
        
        Args:
            stats: Estadísticas descriptivas
            metrics: Métricas de regresión (opcional)
            correlation: Coeficiente de correlación (opcional)
        """
        print("\n" + "="*60)
        print("ANÁLISIS ESTADÍSTICO DE DATOS")
        print("="*60)
        
        print("\n--- Estadísticas Variable X ---")
        for key, value in stats['X'].items():
            print(f"{key.capitalize()}: {value:.4f}")
        
        print("\n--- Estadísticas Variable Y ---")
        for key, value in stats['Y'].items():
            print(f"{key.capitalize()}: {value:.4f}")
        
        if correlation is not None:
            print(f"\n--- Correlación ---")
            print(f"Coeficiente de Pearson: {correlation:.4f}")
            
            # Interpretación de la correlación
            if abs(correlation) >= 0.9:
                interp = "muy fuerte"
            elif abs(correlation) >= 0.7:
                interp = "fuerte"
            elif abs(correlation) >= 0.5:
                interp = "moderada"
            elif abs(correlation) >= 0.3:
                interp = "débil"
            else:
                interp = "muy débil"
            
            direction = "positiva" if correlation > 0 else "negativa"
            print(f"Interpretación: Correlación {interp} {direction}")
        
        if metrics is not None:
            print("\n--- Métricas de Regresión ---")
            for key, value in metrics.items():
                print(f"{key}: {value:.4f}")
        
        print("="*60 + "\n")
