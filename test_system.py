"""
Tests básicos para validar el sistema de regresión lineal
"""
import numpy as np
from data_loader import DataLoader
from regression import LinearRegression
from analysis import DataAnalyzer


def test_data_loader():
    """Test del cargador de datos"""
    print("Testing DataLoader...")
    loader = DataLoader()
    
    # Test generación de datos
    X, y = loader.generate_sample_data(n_samples=50, noise=5.0)
    assert len(X) == 50, "Error: Número incorrecto de muestras"
    assert len(y) == 50, "Error: Número incorrecto de muestras"
    
    # Test resumen de datos
    summary = loader.get_data_summary()
    assert 'n_samples' in summary, "Error: Falta clave n_samples"
    assert summary['n_samples'] == 50, "Error: n_samples incorrecto"
    
    print("✓ DataLoader tests passed")


def test_linear_regression():
    """Test del modelo de regresión lineal"""
    print("Testing LinearRegression...")
    
    # Crear datos lineales simples: y = 3x + 5
    X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    y = np.array([8, 11, 14, 17, 20])  # 3*x + 5
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Verificar coeficientes (deben ser aproximadamente 3 y 5)
    assert abs(model.coef_ - 3.0) < 0.1, f"Error: Coeficiente incorrecto {model.coef_}"
    assert abs(model.intercept_ - 5.0) < 0.1, f"Error: Intercepto incorrecto {model.intercept_}"
    
    # Test predicción
    y_pred = model.predict(X)
    assert len(y_pred) == len(y), "Error: Longitud de predicción incorrecta"
    
    # Test R²
    r2 = model.score(X, y)
    assert r2 > 0.99, f"Error: R² demasiado bajo {r2}"
    
    # Test ecuación
    equation = model.get_equation()
    assert "y =" in equation, "Error: Formato de ecuación incorrecto"
    
    print("✓ LinearRegression tests passed")


def test_data_analyzer():
    """Test del analizador de datos"""
    print("Testing DataAnalyzer...")
    
    X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    y = np.array([2, 4, 6, 8, 10])
    
    analyzer = DataAnalyzer()
    
    # Test estadísticas
    stats = analyzer.calculate_statistics(X, y)
    assert 'X' in stats, "Error: Falta clave X en stats"
    assert 'Y' in stats, "Error: Falta clave Y en stats"
    assert abs(stats['X']['media'] - 3.0) < 0.1, "Error: Media X incorrecta"
    
    # Test correlación
    corr = analyzer.calculate_correlation(X, y)
    assert abs(corr - 1.0) < 0.01, f"Error: Correlación incorrecta {corr}"
    
    # Test métricas
    y_pred = y  # Predicción perfecta
    mse = analyzer.calculate_mse(y, y_pred)
    assert mse == 0.0, f"Error: MSE debería ser 0 {mse}"
    
    mae = analyzer.calculate_mae(y, y_pred)
    assert mae == 0.0, f"Error: MAE debería ser 0 {mae}"
    
    print("✓ DataAnalyzer tests passed")


def test_integration():
    """Test de integración completo"""
    print("Testing Integration...")
    
    # Cargar datos
    loader = DataLoader()
    X, y = loader.generate_sample_data(n_samples=100, noise=10.0)
    
    # Entrenar modelo
    model = LinearRegression()
    model.fit(X, y)
    
    # Predicciones
    y_pred = model.predict(X)
    
    # Análisis
    analyzer = DataAnalyzer()
    r2 = model.score(X, y)
    metrics = analyzer.regression_metrics(y, y_pred, r2)
    
    # Verificaciones
    assert r2 > 0.5, f"Error: R² demasiado bajo {r2}"
    assert 'MSE' in metrics, "Error: Falta MSE en métricas"
    assert 'RMSE' in metrics, "Error: Falta RMSE en métricas"
    assert 'MAE' in metrics, "Error: Falta MAE en métricas"
    
    print("✓ Integration tests passed")


def run_all_tests():
    """Ejecuta todos los tests"""
    print("\n" + "="*60)
    print("EJECUTANDO TESTS DEL SISTEMA")
    print("="*60 + "\n")
    
    try:
        test_data_loader()
        test_linear_regression()
        test_data_analyzer()
        test_integration()
        
        print("\n" + "="*60)
        print("✓ TODOS LOS TESTS PASARON EXITOSAMENTE")
        print("="*60 + "\n")
        return True
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
