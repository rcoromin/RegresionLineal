"""
Módulo para cargar y minar datos de diferentes fuentes
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional


class DataLoader:
    """Clase para cargar y procesar datos"""
    
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
    
    def load_csv(self, filepath: str, x_column: str, y_column: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Carga datos desde un archivo CSV
        
        Args:
            filepath: Ruta al archivo CSV
            x_column: Nombre de la columna para variable independiente
            y_column: Nombre de la columna para variable dependiente
            
        Returns:
            Tupla con arrays X e y
        """
        self.data = pd.read_csv(filepath)
        self.X = self.data[x_column].values.reshape(-1, 1)
        self.y = self.data[y_column].values
        return self.X, self.y
    
    def load_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Carga datos desde arrays numpy
        
        Args:
            X: Array con variable independiente
            y: Array con variable dependiente
            
        Returns:
            Tupla con arrays X e y procesados
        """
        self.X = X.reshape(-1, 1) if X.ndim == 1 else X
        self.y = y
        return self.X, self.y
    
    def generate_sample_data(self, n_samples: int = 100, noise: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Genera datos de muestra para demostración
        
        Args:
            n_samples: Número de muestras a generar
            noise: Nivel de ruido a agregar
            
        Returns:
            Tupla con arrays X e y generados
        """
        np.random.seed(42)
        self.X = np.random.rand(n_samples, 1) * 100
        self.y = 2.5 * self.X.flatten() + 30 + np.random.randn(n_samples) * noise
        return self.X, self.y
    
    def get_data_summary(self) -> dict:
        """
        Obtiene un resumen estadístico de los datos cargados
        
        Returns:
            Diccionario con estadísticas básicas
        """
        if self.X is None or self.y is None:
            return {}
        
        return {
            'n_samples': len(self.X),
            'X_mean': float(np.mean(self.X)),
            'X_std': float(np.std(self.X)),
            'X_min': float(np.min(self.X)),
            'X_max': float(np.max(self.X)),
            'y_mean': float(np.mean(self.y)),
            'y_std': float(np.std(self.y)),
            'y_min': float(np.min(self.y)),
            'y_max': float(np.max(self.y))
        }
