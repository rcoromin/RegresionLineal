"""
Módulo de visualización y gráficos
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


class Visualizer:
    """Clase para crear visualizaciones de datos y regresión"""
    
    def __init__(self, figsize: tuple = (10, 6)):
        self.figsize = figsize
    
    def plot_data(self, X: np.ndarray, y: np.ndarray, 
                  title: str = "Datos", 
                  xlabel: str = "X", 
                  ylabel: str = "Y",
                  save_path: Optional[str] = None):
        """
        Grafica los datos originales
        
        Args:
            X: Variable independiente
            y: Variable dependiente
            title: Título del gráfico
            xlabel: Etiqueta del eje X
            ylabel: Etiqueta del eje Y
            save_path: Ruta para guardar el gráfico (opcional)
        """
        plt.figure(figsize=self.figsize)
        plt.scatter(X, y, alpha=0.6, color='blue', edgecolors='k', s=50)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_regression(self, X: np.ndarray, y: np.ndarray, 
                       y_pred: np.ndarray,
                       equation: str = "",
                       r2_score: float = 0.0,
                       title: str = "Regresión Lineal",
                       xlabel: str = "X",
                       ylabel: str = "Y",
                       save_path: Optional[str] = None):
        """
        Grafica datos originales con línea de regresión
        
        Args:
            X: Variable independiente
            y: Variable dependiente (datos reales)
            y_pred: Predicciones del modelo
            equation: Ecuación de la regresión
            r2_score: Coeficiente R²
            title: Título del gráfico
            xlabel: Etiqueta del eje X
            ylabel: Etiqueta del eje Y
            save_path: Ruta para guardar el gráfico (opcional)
        """
        plt.figure(figsize=self.figsize)
        
        # Graficar datos originales
        plt.scatter(X, y, alpha=0.6, color='blue', 
                   edgecolors='k', s=50, label='Datos reales')
        
        # Ordenar X para graficar línea correctamente
        sort_idx = np.argsort(X.flatten())
        X_sorted = X.flatten()[sort_idx]
        y_pred_sorted = y_pred[sort_idx]
        
        # Graficar línea de regresión
        plt.plot(X_sorted, y_pred_sorted, 
                color='red', linewidth=2, label='Línea de regresión')
        
        # Agregar información de la ecuación y R²
        textstr = f'{equation}\nR² = {r2_score:.4f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top', bbox=props)
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_residuals(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray,
                      title: str = "Análisis de Residuos",
                      save_path: Optional[str] = None):
        """
        Grafica los residuos (errores de predicción)
        
        Args:
            X: Variable independiente
            y: Variable dependiente (datos reales)
            y_pred: Predicciones del modelo
            title: Título del gráfico
            save_path: Ruta para guardar el gráfico (opcional)
        """
        residuals = y - y_pred
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Gráfico de residuos vs X
        ax1.scatter(X, residuals, alpha=0.6, color='purple', edgecolors='k', s=50)
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax1.set_title('Residuos vs X', fontsize=12, fontweight='bold')
        ax1.set_xlabel('X', fontsize=10)
        ax1.set_ylabel('Residuos', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Histograma de residuos
        ax2.hist(residuals, bins=20, alpha=0.7, color='purple', edgecolor='k')
        ax2.set_title('Distribución de Residuos', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Residuos', fontsize=10)
        ax2.set_ylabel('Frecuencia', fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
