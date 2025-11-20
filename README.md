# Sistema de Minería de Datos y Regresión Lineal

![Captura del Sistema](Sistema%20mineria%20Datos.png)

Este proyecto es una aplicación de escritorio desarrollada en Python que permite realizar análisis de minería de datos sobre conjuntos de datos inmobiliarios. Utiliza una interfaz gráfica (GUI) construida con `tkinter` para facilitar la carga, limpieza, análisis y visualización de datos.

## 🚀 Funcionalidades Principales

1.  **Carga de Datos:**
    *   Soporte para archivos Excel (`.xlsx`, `.xlsm`).
    *   Normalización automática de columnas (detecta y renombra 'precio_usd' y 'metros_cuad').
    *   Generación automática de un **Resumen Estadístico** y un **Glosario de Términos** dinámico en los logs.

2.  **Limpieza de Datos Inteligente:**
    *   Eliminación de valores nulos y ceros ilógicos.
    *   **Filtrado de Outliers (Ruido):** Elimina automáticamente propiedades con superficies excesivas (> 2,000 m²) y precios absurdos (> 100 Millones USD) para evitar distorsiones en el modelo.

3.  **Regresión Lineal Simple:**
    *   Entrenamiento de un modelo de Machine Learning (`LinearRegression`) para predecir el valor de una propiedad en función de su superficie.
    *   Cálculo y visualización de la ecuación de la recta ($y = mx + b$).
    *   Gráfico de dispersión con la línea de tendencia.

4.  **Pronóstico Interactivo:**
    *   Permite al usuario ingresar múltiples superficies manualmente.
    *   Calcula el precio estimado para cada una.
    *   **Visualización en tiempo real:** Actualiza el gráfico destacando los nuevos puntos pronosticados con marcadores verdes ("X") sobre la línea de regresión.

5.  **Agrupamiento (Clustering):**
    *   Implementación del algoritmo **K-Means**.
    *   Agrupa las propiedades en 3 clusters según similitud de precio y superficie.
    *   Visualización de grupos con mapa de colores y centroides.

## 🛠️ Tecnologías Utilizadas

*   **Lenguaje:** Python 3.13+
*   **Interfaz Gráfica:** `tkinter` (Nativa de Python)
*   **Manipulación de Datos:** `pandas`, `numpy`
*   **Visualización:** `matplotlib` (Integrado en Tkinter)
*   **Machine Learning:** `scikit-learn`

## 📋 Requisitos de Instalación

Asegúrate de tener Python instalado. Se recomienda usar un entorno virtual.

1.  **Clonar el repositorio:**
    ```bash
    git clone https://github.com/rcoromin/RegresionLineal.git
    cd RegresionLineal
    ```

2.  **Instalar dependencias:**
    ```bash
    pip install pandas numpy matplotlib scikit-learn openpyxl
    ```

## ▶️ Cómo Ejecutar

Ejecuta el archivo principal desde tu terminal:

```bash
python Mineria.py
```

## 📂 Estructura del Proyecto

*   `Mineria.py`: Código fuente principal de la aplicación.
*   `Propiedades_Precios.xlsm`: Dataset de ejemplo (si está disponible).
*   `README.md`: Documentación del proyecto.

---
Desarrollado para la asignatura de Minería de Datos.
