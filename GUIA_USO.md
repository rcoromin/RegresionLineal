# Guía de Uso - Sistema de Regresión Lineal

## 📚 Introducción

Este sistema proporciona herramientas completas para análisis de regresión lineal y minería de datos. Incluye carga de datos, análisis estadístico, modelado y visualización.

## 🚀 Inicio Rápido

### 1. Instalación

```bash
pip install -r requirements.txt
```

### 2. Uso Básico

#### Ejecutar el ejemplo completo:
```bash
python main.py
```

Esto generará:
- Análisis estadístico completo
- Modelo de regresión lineal entrenado
- Gráficos guardados como PNG
- Métricas de evaluación

#### Usar con tus propios datos CSV:
```bash
python example_csv.py
```

## 📖 Ejemplos Detallados

### Ejemplo 1: Cargar y Analizar Datos CSV

```python
from data_loader import DataLoader
from regression import LinearRegression
from analysis import DataAnalyzer

# Cargar datos
loader = DataLoader()
X, y = loader.load_csv('mi_archivo.csv', 'columna_x', 'columna_y')

# Obtener resumen
summary = loader.get_data_summary()
print(f"Número de muestras: {summary['n_samples']}")
print(f"Media X: {summary['X_mean']:.2f}")
```

### Ejemplo 2: Entrenar Modelo de Regresión

```python
from regression import LinearRegression

# Crear y entrenar modelo
model = LinearRegression()
model.fit(X, y)

# Obtener ecuación
print(model.get_equation())
# Output: y = 2.5000x +10.0000

# Calcular R²
r2 = model.score(X, y)
print(f"R² = {r2:.4f}")
```

### Ejemplo 3: Hacer Predicciones

```python
import numpy as np

# Predecir con nuevos valores
nuevos_valores = np.array([15, 25, 35, 45])
predicciones = model.predict(nuevos_valores)

for x, y_pred in zip(nuevos_valores, predicciones):
    print(f"X = {x} -> Y predicho = {y_pred:.2f}")
```

### Ejemplo 4: Análisis Estadístico Completo

```python
from analysis import DataAnalyzer

analyzer = DataAnalyzer()

# Estadísticas descriptivas
stats = analyzer.calculate_statistics(X, y)

# Correlación
correlation = analyzer.calculate_correlation(X, y)
print(f"Correlación: {correlation:.4f}")

# Métricas de regresión
y_pred = model.predict(X)
metrics = analyzer.regression_metrics(y, y_pred, r2)

print(f"MSE: {metrics['MSE']:.4f}")
print(f"RMSE: {metrics['RMSE']:.4f}")
print(f"MAE: {metrics['MAE']:.4f}")
```

### Ejemplo 5: Crear Visualizaciones

```python
from visualization import Visualizer

visualizer = Visualizer(figsize=(12, 6))

# Gráfico de regresión
visualizer.plot_regression(
    X, y, y_pred,
    equation=model.get_equation(),
    r2_score=r2,
    title="Mi Análisis de Regresión",
    xlabel="Variable X",
    ylabel="Variable Y",
    save_path="mi_grafico.png"
)

# Análisis de residuos
visualizer.plot_residuals(X, y, y_pred, 
                         save_path="residuos.png")
```

### Ejemplo 6: Generar Datos de Prueba

```python
from data_loader import DataLoader

loader = DataLoader()

# Generar datos sintéticos
X, y = loader.generate_sample_data(
    n_samples=200,  # Número de puntos
    noise=20.0      # Nivel de ruido
)
```

## 🔍 Interpretación de Resultados

### Coeficiente R² (R-cuadrado)
- **0.9 - 1.0**: Excelente ajuste
- **0.7 - 0.9**: Buen ajuste
- **0.5 - 0.7**: Ajuste moderado
- **< 0.5**: Ajuste pobre

### Correlación de Pearson
- **±0.9 - ±1.0**: Correlación muy fuerte
- **±0.7 - ±0.9**: Correlación fuerte
- **±0.5 - ±0.7**: Correlación moderada
- **±0.3 - ±0.5**: Correlación débil
- **< ±0.3**: Correlación muy débil

### Métricas de Error
- **MSE** (Error Cuadrático Medio): Penaliza errores grandes
- **RMSE** (Raíz del MSE): Mismo orden de magnitud que los datos
- **MAE** (Error Absoluto Medio): Promedio de errores absolutos

## 📊 Formato de Datos CSV

Tu archivo CSV debe tener al menos dos columnas:

```csv
Variable_X,Variable_Y
10,25
20,45
30,65
40,85
50,105
```

## 🎨 Personalización de Gráficos

```python
# Cambiar tamaño de figura
visualizer = Visualizer(figsize=(14, 8))

# Personalizar títulos y etiquetas
visualizer.plot_regression(
    X, y, y_pred,
    equation=model.get_equation(),
    r2_score=r2,
    title="Análisis Personalizado",
    xlabel="Temperatura (°C)",
    ylabel="Ventas ($)",
    save_path="ventas_vs_temperatura.png"
)
```

## 🧪 Ejecutar Tests

```bash
python test_system.py
```

Esto validará:
- Carga de datos
- Regresión lineal
- Análisis estadístico
- Integración completa

## 💡 Consejos

1. **Inspecciona tus datos primero**: Usa `get_data_summary()` antes del análisis
2. **Verifica la correlación**: Una correlación débil indica que regresión lineal puede no ser apropiada
3. **Analiza los residuos**: Deben estar distribuidos aleatoriamente alrededor de cero
4. **Guarda tus gráficos**: Usa el parámetro `save_path` para documentar resultados

## 🐛 Solución de Problemas

### Error: "El modelo debe ser entrenado primero"
```python
# Asegúrate de llamar fit() antes de predict()
model.fit(X, y)
y_pred = model.predict(X)
```

### Error al cargar CSV
```python
# Verifica que los nombres de columnas sean correctos
loader.load_csv('datos.csv', 'X', 'Y')  # Nombres exactos
```

### Gráficos no se muestran
```python
# En entornos sin display, solo guarda la imagen
visualizer.plot_regression(..., save_path="output.png")
```

## 📞 Soporte

Para reportar problemas o sugerir mejoras, abre un issue en el repositorio de GitHub.
