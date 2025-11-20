"""
Ejemplo avanzado de uso del sistema de regresión lineal
Muestra características avanzadas y flujo de trabajo completo
"""
import numpy as np
from data_loader import DataLoader
from regression import LinearRegression
from visualization import Visualizer
from analysis import DataAnalyzer


def ejemplo_completo():
    """Ejemplo de flujo de trabajo completo"""
    print("\n" + "="*70)
    print("EJEMPLO AVANZADO - ANÁLISIS DE VENTAS VS TEMPERATURA")
    print("="*70 + "\n")
    
    # Simulación de datos de ventas de helados vs temperatura
    print("[1] Generando datos de ejemplo (Ventas de helados vs Temperatura)...")
    
    # Crear datos realistas: ventas aumentan con temperatura
    np.random.seed(123)
    temperatura = np.random.uniform(15, 35, 80)  # Temperatura en °C
    # Ventas base + efecto temperatura + ruido
    ventas = 50 + 3.5 * temperatura + np.random.normal(0, 15, 80)
    
    # Cargar datos
    loader = DataLoader()
    X, y = loader.load_data(temperatura, ventas)
    
    print(f"✓ Dataset creado: {len(X)} días de datos")
    print(f"  - Temperatura: {temperatura.min():.1f}°C a {temperatura.max():.1f}°C")
    print(f"  - Ventas: ${ventas.min():.2f} a ${ventas.max():.2f}")
    
    # Análisis exploratorio
    print("\n[2] Análisis exploratorio de datos...")
    analyzer = DataAnalyzer()
    
    stats = analyzer.calculate_statistics(X, y)
    print(f"  - Temperatura media: {stats['X']['media']:.2f}°C")
    print(f"  - Ventas promedio: ${stats['Y']['media']:.2f}")
    print(f"  - Desviación std ventas: ${stats['Y']['desviacion_std']:.2f}")
    
    correlation = analyzer.calculate_correlation(X, y)
    print(f"  - Correlación: {correlation:.4f}")
    
    if abs(correlation) > 0.7:
        print("  → Correlación fuerte detectada: Regresión lineal es apropiada ✓")
    else:
        print("  ⚠ Advertencia: Correlación débil, considerar otros modelos")
    
    # Entrenar modelo
    print("\n[3] Entrenando modelo de regresión...")
    model = LinearRegression()
    model.fit(X, y)
    
    equation = model.get_equation()
    print(f"✓ Modelo entrenado: {equation}")
    print(f"  Interpretación: Por cada grado de temperatura,")
    print(f"  las ventas aumentan en ${model.coef_:.2f}")
    
    # Predicciones
    print("\n[4] Realizando predicciones...")
    y_pred = model.predict(X)
    r2 = model.score(X, y)
    
    print(f"✓ Coeficiente R²: {r2:.4f}")
    
    # Métricas de error
    metrics = analyzer.regression_metrics(y, y_pred, r2)
    print(f"  - Error promedio (MAE): ${metrics['MAE']:.2f}")
    print(f"  - Error cuadrático medio (RMSE): ${metrics['RMSE']:.2f}")
    
    # Visualizaciones
    print("\n[5] Generando visualizaciones...")
    visualizer = Visualizer(figsize=(14, 6))
    
    visualizer.plot_regression(
        X, y, y_pred,
        equation=equation,
        r2_score=r2,
        title="Análisis de Ventas de Helados vs Temperatura",
        xlabel="Temperatura (°C)",
        ylabel="Ventas ($)",
        save_path="ventas_temperatura.png"
    )
    print("✓ Gráfico guardado: ventas_temperatura.png")
    
    visualizer.plot_residuals(
        X, y, y_pred,
        title="Análisis de Residuos - Ventas vs Temperatura",
        save_path="residuos_ventas.png"
    )
    print("✓ Análisis de residuos guardado: residuos_ventas.png")
    
    # Predicciones para planificación
    print("\n[6] Predicciones para planificación:")
    temperaturas_futuras = np.array([20, 25, 30, 35])
    ventas_predichas = model.predict(temperaturas_futuras)
    
    print("  Pronóstico de ventas:")
    for temp, venta in zip(temperaturas_futuras, ventas_predichas):
        print(f"    {temp}°C → ${venta:.2f} en ventas esperadas")
    
    # Análisis de sensibilidad
    print("\n[7] Análisis de sensibilidad:")
    incremento_temp = 5  # °C
    incremento_ventas = model.coef_ * incremento_temp
    print(f"  Un aumento de {incremento_temp}°C en temperatura")
    print(f"  resulta en ${incremento_ventas:.2f} más en ventas")
    
    # Intervalos de confianza (simplificado)
    print("\n[8] Evaluación de calidad del modelo:")
    residuos = y - y_pred
    std_residuos = np.std(residuos)
    
    print(f"  - Desviación estándar de residuos: ${std_residuos:.2f}")
    print(f"  - 68% de predicciones dentro de ±${std_residuos:.2f}")
    print(f"  - 95% de predicciones dentro de ±${2*std_residuos:.2f}")
    
    # Análisis completo
    print("\n[9] Resumen del análisis:")
    analyzer.print_analysis(stats, metrics, correlation)
    
    print("\n" + "="*70)
    print("ANÁLISIS COMPLETADO")
    print("="*70)
    print("\nConclusiones:")
    print("✓ Modelo de regresión lineal válido para predicción de ventas")
    print("✓ Relación positiva fuerte entre temperatura y ventas")
    print("✓ Gráficos y métricas guardados para reportes")
    print("\nArchivos generados:")
    print("  - ventas_temperatura.png")
    print("  - residuos_ventas.png")
    print("="*70 + "\n")


def ejemplo_comparacion_modelos():
    """Compara múltiples conjuntos de datos"""
    print("\n" + "="*70)
    print("EJEMPLO: COMPARACIÓN DE MÚLTIPLES DATASETS")
    print("="*70 + "\n")
    
    np.random.seed(456)
    
    datasets = {
        "Datos con poco ruido": (
            np.random.rand(50, 1) * 100,
            2.5 * np.random.rand(50, 1).flatten() * 100 + 30 + np.random.randn(50) * 5
        ),
        "Datos con ruido moderado": (
            np.random.rand(50, 1) * 100,
            2.5 * np.random.rand(50, 1).flatten() * 100 + 30 + np.random.randn(50) * 20
        ),
        "Datos con mucho ruido": (
            np.random.rand(50, 1) * 100,
            2.5 * np.random.rand(50, 1).flatten() * 100 + 30 + np.random.randn(50) * 40
        )
    }
    
    print("Comparando calidad de ajuste en diferentes niveles de ruido:\n")
    print(f"{'Dataset':<30} {'R²':<10} {'RMSE':<10} {'Correlación':<15}")
    print("-" * 70)
    
    analyzer = DataAnalyzer()
    
    for nombre, (X, y) in datasets.items():
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        r2 = model.score(X, y)
        rmse = analyzer.calculate_rmse(y, y_pred)
        corr = analyzer.calculate_correlation(X, y)
        
        print(f"{nombre:<30} {r2:<10.4f} {rmse:<10.2f} {corr:<15.4f}")
    
    print("-" * 70)
    print("\nObservación: El ruido afecta R² y RMSE pero la correlación se mantiene.\n")


if __name__ == "__main__":
    ejemplo_completo()
    ejemplo_comparacion_modelos()
