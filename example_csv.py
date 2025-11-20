"""
Ejemplo de uso con datos CSV
"""
from data_loader import DataLoader
from regression import LinearRegression
from visualization import Visualizer
from analysis import DataAnalyzer


def main():
    """Ejemplo de análisis con archivo CSV"""
    
    print("\n" + "="*70)
    print("EJEMPLO: ANÁLISIS DE DATOS DESDE CSV")
    print("="*70)
    
    # Cargar datos desde CSV
    print("\n[1] Cargando datos desde sample_data.csv...")
    loader = DataLoader()
    X, y = loader.load_csv('sample_data.csv', 'X', 'Y')
    print(f"✓ Datos cargados: {len(X)} registros")
    
    # Análisis estadístico
    print("\n[2] Análisis estadístico...")
    analyzer = DataAnalyzer()
    stats = analyzer.calculate_statistics(X, y)
    correlation = analyzer.calculate_correlation(X, y)
    
    # Entrenar modelo
    print("\n[3] Entrenando modelo...")
    model = LinearRegression()
    model.fit(X, y)
    equation = model.get_equation()
    print(f"✓ {equation}")
    
    # Predicciones y evaluación
    y_pred = model.predict(X)
    r2 = model.score(X, y)
    metrics = analyzer.regression_metrics(y, y_pred, r2)
    
    # Visualización
    print("\n[4] Generando visualización...")
    visualizer = Visualizer()
    visualizer.plot_regression(
        X, y, y_pred,
        equation=equation,
        r2_score=r2,
        title="Regresión Lineal - Datos CSV",
        save_path="csv_regression.png"
    )
    
    # Imprimir análisis
    analyzer.print_analysis(stats, metrics, correlation)
    
    print("✓ Análisis completado. Gráfico guardado en: csv_regression.png\n")


if __name__ == "__main__":
    main()
