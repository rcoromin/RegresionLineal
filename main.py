"""
Sistema de Minería de Datos y Regresión Lineal
Aplicación principal
"""
import numpy as np
from data_loader import DataLoader
from regression import LinearRegression
from visualization import Visualizer
from analysis import DataAnalyzer


def main():
    """Función principal del sistema"""
    
    print("\n" + "="*70)
    print("SISTEMA DE MINERÍA DE DATOS Y REGRESIÓN LINEAL")
    print("="*70)
    
    # 1. Cargar datos
    print("\n[1] Cargando datos...")
    loader = DataLoader()
    
    # Generar datos de ejemplo (se puede reemplazar con load_csv)
    X, y = loader.generate_sample_data(n_samples=100, noise=15.0)
    print(f"✓ Datos cargados: {len(X)} muestras")
    
    # Resumen de datos
    summary = loader.get_data_summary()
    print(f"  - Rango X: [{summary['X_min']:.2f}, {summary['X_max']:.2f}]")
    print(f"  - Rango Y: [{summary['y_min']:.2f}, {summary['y_max']:.2f}]")
    
    # 2. Análisis estadístico
    print("\n[2] Realizando análisis estadístico...")
    analyzer = DataAnalyzer()
    
    # Calcular estadísticas descriptivas
    stats = analyzer.calculate_statistics(X, y)
    
    # Calcular correlación
    correlation = analyzer.calculate_correlation(X, y)
    
    print(f"✓ Correlación de Pearson: {correlation:.4f}")
    
    # 3. Entrenar modelo de regresión lineal
    print("\n[3] Entrenando modelo de regresión lineal...")
    model = LinearRegression()
    model.fit(X, y)
    
    # Obtener ecuación
    equation = model.get_equation()
    print(f"✓ Ecuación: {equation}")
    
    # 4. Realizar predicciones
    print("\n[4] Realizando predicciones...")
    y_pred = model.predict(X)
    
    # Calcular R²
    r2 = model.score(X, y)
    print(f"✓ Coeficiente R²: {r2:.4f}")
    
    # 5. Calcular métricas de evaluación
    print("\n[5] Calculando métricas de evaluación...")
    metrics = analyzer.regression_metrics(y, y_pred, r2)
    print(f"✓ MSE: {metrics['MSE']:.4f}")
    print(f"✓ RMSE: {metrics['RMSE']:.4f}")
    print(f"✓ MAE: {metrics['MAE']:.4f}")
    
    # 6. Visualización
    print("\n[6] Generando visualizaciones...")
    visualizer = Visualizer(figsize=(12, 6))
    
    # Gráfico de regresión
    print("  - Gráfico de regresión lineal")
    visualizer.plot_regression(
        X, y, y_pred,
        equation=equation,
        r2_score=r2,
        title="Regresión Lineal - Análisis de Datos",
        xlabel="Variable Independiente (X)",
        ylabel="Variable Dependiente (Y)",
        save_path="regression_plot.png"
    )
    
    # Análisis de residuos
    print("  - Análisis de residuos")
    visualizer.plot_residuals(
        X, y, y_pred,
        title="Análisis de Residuos",
        save_path="residuals_plot.png"
    )
    
    # 7. Imprimir análisis completo
    analyzer.print_analysis(stats, metrics, correlation)
    
    print("\n[7] ✓ Análisis completado exitosamente")
    print(f"  - Gráficos guardados: regression_plot.png, residuals_plot.png")
    
    # 8. Ejemplo de predicción con nuevos valores
    print("\n[8] Ejemplo de predicción con nuevos valores:")
    new_X = np.array([25, 50, 75])
    new_predictions = model.predict(new_X)
    
    for x_val, y_val in zip(new_X, new_predictions):
        print(f"  X = {x_val:.2f} -> Y predicho = {y_val:.2f}")
    
    print("\n" + "="*70)
    print("FIN DEL ANÁLISIS")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
