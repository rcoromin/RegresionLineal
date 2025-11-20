# Sistema de Regresión Lineal y Minería de Datos

Sistema completo para minar datos y realizar análisis de regresión lineal con visualizaciones gráficas.

## 📋 Características

- **Carga de Datos**: Soporte para archivos CSV y generación de datos de muestra
- **Regresión Lineal**: Implementación desde cero del algoritmo de regresión lineal
- **Análisis Estadístico**: Cálculo de estadísticas descriptivas, correlación y métricas de evaluación
- **Visualizaciones**: Gráficos de dispersión, líneas de regresión y análisis de residuos
- **Métricas de Evaluación**: R², MSE, RMSE, MAE

## 🚀 Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/rcoromin/RegresionLineal.git
cd RegresionLineal
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## 📦 Dependencias

- numpy >= 1.21.0
- matplotlib >= 3.5.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0

## 💻 Uso

### Ejemplo Básico

Ejecutar el análisis con datos generados automáticamente:

```bash
python main.py
```

Este comando:
1. Genera datos de muestra
2. Realiza análisis estadístico completo
3. Entrena un modelo de regresión lineal
4. Genera visualizaciones (gráficos guardados como PNG)
5. Muestra métricas de evaluación

### Usar con Archivo CSV

```bash
python example_csv.py
```

O usar la biblioteca en tu propio código:

```python
from data_loader import DataLoader
from regression import LinearRegression
from visualization import Visualizer
from analysis import DataAnalyzer

# Cargar datos desde CSV
loader = DataLoader()
X, y = loader.load_csv('datos.csv', 'columna_x', 'columna_y')

# Entrenar modelo
model = LinearRegression()
model.fit(X, y)

# Hacer predicciones
y_pred = model.predict(X)

# Visualizar resultados
visualizer = Visualizer()
visualizer.plot_regression(X, y, y_pred, 
                          equation=model.get_equation(),
                          r2_score=model.score(X, y))
```

## 📊 Estructura del Proyecto

```
RegresionLineal/
├── data_loader.py      # Módulo de carga y minería de datos
├── regression.py       # Implementación de regresión lineal
├── visualization.py    # Módulo de visualización
├── analysis.py         # Análisis estadístico y métricas
├── main.py            # Aplicación principal
├── example_csv.py     # Ejemplo con datos CSV
├── sample_data.csv    # Datos de ejemplo
├── requirements.txt   # Dependencias
└── README.md          # Este archivo
```

## 📈 Módulos

### data_loader.py
- `DataLoader`: Clase para cargar datos desde CSV, arrays o generar datos de muestra
- Métodos: `load_csv()`, `load_data()`, `generate_sample_data()`, `get_data_summary()`

### regression.py
- `LinearRegression`: Implementación de regresión lineal simple
- Métodos: `fit()`, `predict()`, `score()`, `get_equation()`

### visualization.py
- `Visualizer`: Clase para crear visualizaciones
- Métodos: `plot_data()`, `plot_regression()`, `plot_residuals()`

### analysis.py
- `DataAnalyzer`: Análisis estadístico completo
- Métodos: `calculate_statistics()`, `calculate_correlation()`, `regression_metrics()`

## 📝 Ejemplo de Salida

El sistema genera:

1. **Estadísticas Descriptivas**: Media, mediana, desviación estándar, varianza, etc.
2. **Ecuación de Regresión**: y = mx + b
3. **Coeficiente R²**: Calidad del ajuste del modelo
4. **Métricas de Error**: MSE, RMSE, MAE
5. **Gráficos**: 
   - Dispersión de datos con línea de regresión
   - Análisis de residuos

## 🎯 Características del Análisis

- Correlación de Pearson entre variables
- Coeficiente de determinación (R²)
- Error Cuadrático Medio (MSE)
- Raíz del Error Cuadrático Medio (RMSE)
- Error Absoluto Medio (MAE)
- Visualización de residuos
- Distribución de errores

## 🔧 Formato de Datos CSV

Los archivos CSV deben tener al menos dos columnas:

```csv
X,Y
10.5,55.2
20.3,78.5
30.1,102.3
...
```

## 📄 Licencia

Este proyecto está bajo licencia MIT.

## 👤 Autor

rcoromin
