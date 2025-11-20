# --- Librerías para la Ejecución del Sistema e Interfaz Gráfica ---
import os  # Librería para interactuar con el sistema operativo (rutas de archivos, verificar existencia)
import tkinter as tk  # Librería estándar de Python para crear interfaces gráficas (GUI)
from tkinter import ttk, filedialog, messagebox, scrolledtext  # Widgets avanzados y diálogos de Tkinter

# --- Librerías para Minería de Datos y Visualización ---
import pandas as pd  # Librería para manipulación y análisis de datos (DataFrames)
import numpy as np   # Librería para operaciones matemáticas y manejo de arrays numéricos
import matplotlib.pyplot as plt  # Librería para la creación de gráficos y visualizaciones
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Permite incrustar gráficos de Matplotlib en ventanas de Tkinter
from sklearn.linear_model import LinearRegression  # Algoritmo de Machine Learning para Regresión Lineal
from sklearn.cluster import KMeans  # Algoritmo de Machine Learning para Agrupamiento (Clustering)

class SistemaMineriaGUI:
    """
    Clase principal que gestiona la interfaz gráfica y la lógica de minería de datos.
    Utiliza tkinter para la UI y pandas/scikit-learn para el procesamiento de datos.
    """
    def __init__(self, root):
        """
        Constructor de la clase.
        Inicializa la ventana principal, variables de estado y configura la interfaz.
        """
        self.root = root
        self.root.title("Sistema de Minería de Datos")
        self.root.geometry("1200x800")

        # Configuración global de pandas para formato de números (sin notación científica, con separador de miles)
        pd.set_option('display.float_format', lambda x: '{:,.2f}'.format(x))

        # Variables para almacenar el estado de los datos y modelos
        self.df = None          # DataFrame original cargado
        self.df_clean = None    # DataFrame después de la limpieza
        self.modelo = None      # Modelo de Regresión Lineal entrenado
        self.kmeans = None      # Modelo de Clustering K-Means entrenado

        # --- UI Setup ---
        self.setup_ui()

    def setup_ui(self):
        """
        Configura y distribuye todos los elementos visuales de la interfaz gráfica.
        Divide la ventana en paneles: Control (Izquierda), Contenido (Derecha) y Logs (Abajo).
        """
        # Contenedor principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Panel Izquierdo: Controles
        control_frame = ttk.LabelFrame(main_frame, text="Menú de Operaciones", padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        ttk.Button(control_frame, text="1. Cargar Datos", command=self.cargar_datos).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="2. Limpiar Datos", command=self.limpiar_datos).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="3. Regresión Lineal", command=self.regresion_lineal).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="4. Pronóstico", command=self.pronostico_dialog).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="5. Agrupamiento (K-Means)", command=self.agrupamiento).pack(fill=tk.X, pady=5)
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Button(control_frame, text="Salir", command=self.root.quit).pack(fill=tk.X, pady=5)

        # Panel Derecho: Contenido
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Pestañas para Datos y Gráficos
        self.notebook = ttk.Notebook(content_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Pestaña 1: Vista de Datos
        self.tab_data = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_data, text="Vista de Datos")
        
        # Tabla (Treeview) para el dataframe
        self.tree = ttk.Treeview(self.tab_data)
        
        # Scrollbar vertical para la tabla
        scrollbar = ttk.Scrollbar(self.tab_data, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Pestaña 2: Gráficos
        self.tab_graphs = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_graphs, text="Gráficos")
        self.graph_frame = ttk.Frame(self.tab_graphs)
        self.graph_frame.pack(fill=tk.BOTH, expand=True)

        # Panel Inferior: Logs
        log_frame = ttk.LabelFrame(content_frame, text="Detalle del Proceso / Logs", padding="5")
        log_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Usamos fuente 'Consolas' (monoespaciada) para que las tablas y alineaciones se vean bien
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, state='disabled', font=("Consolas", 10))
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def log(self, message):
        """Agrega un mensaje al panel de logs en la parte inferior de la ventana."""
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')

    def update_treeview(self, dataframe):
        """Actualiza la tabla visual (Treeview) con los datos de un DataFrame."""
        # Limpiar existente
        self.tree.delete(*self.tree.get_children())
        
        # Definir columnas incluyendo una para la secuencia (N°)
        cols = ["N°"] + list(dataframe.columns)
        self.tree["columns"] = cols
        self.tree["show"] = "headings"
        
        # Configurar columna de secuencia
        self.tree.heading("N°", text="N°")
        self.tree.column("N°", width=50, anchor="center")
        
        # Configurar resto de columnas
        for col in dataframe.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, anchor="center")

        # Insertar datos con número de secuencia
        for i, (index, row) in enumerate(dataframe.iterrows(), start=1):
            values = [i] + list(row)
            self.tree.insert("", "end", values=values)

    def clear_graph(self):
        """Limpia el área de gráficos antes de dibujar uno nuevo."""
        for widget in self.graph_frame.winfo_children():
            widget.destroy()

    def cargar_datos(self):
        """
        Paso 1: Carga de Datos.
        Intenta cargar 'Propiedades_Precios.xlsm' automáticamente.
        Si no existe, permite al usuario seleccionar un archivo.
        Si no se selecciona nada, carga datos de prueba.
        """
        self.log("--- 1. CARGA DE DATOS ---")
        # Intentar cargar archivo por defecto primero si existe, sino abrir dialogo
        archivo_defecto = 'Propiedades_Precios.xlsm'
        archivo = None
        
        if os.path.exists(archivo_defecto):
             respuesta = messagebox.askyesno("Archivo encontrado", f"Se encontró '{archivo_defecto}'. ¿Desea cargarlo?")
             if respuesta:
                 archivo = archivo_defecto
        
        if not archivo:
            archivo = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xlsm *.xls")])
        
        if archivo:
            try:
                self.df = pd.read_excel(archivo)
                self.log(f"✅ Archivo '{os.path.basename(archivo)}' cargado exitosamente.")
                
                if 'precio_usd' in self.df.columns and 'metros_cuad' in self.df.columns:
                    self.df.rename(columns={'precio_usd': 'Valor', 'metros_cuad': 'Superficie'}, inplace=True)
                    self.log("ℹ️  Se han normalizado los nombres de las columnas a 'Valor' y 'Superficie'.")
            except Exception as e:
                self.log(f"❌ Error al cargar el archivo: {e}")
                return
        else:
            self.log("⚠️  No se seleccionó archivo. Cargando datos de prueba...")
            data = {
                'Superficie': [50, 60, 0, 80, 100, 120, np.nan, 150, 200, 45, 300, 10000],
                'Valor': [2500, 2800, 100, 3500, 4200, 4900, 3000, 6000, 7800, 2100, 11000, 500]
            }
            self.df = pd.DataFrame(data)
        
        self.update_treeview(self.df)
        
        # Calcular estadísticas de la columna 'Valor' para mostrarlas en el glosario
        stats = self.df['Valor'].describe()

        self.log("📊 Estadísticas iniciales (Resumen Estadístico - Columna 'Valor'):")
        self.log("Glosario de términos:")
        self.log(f" • count: ({int(stats['count']):,}) Cantidad total de registros (filas) con datos.")
        self.log(f" • mean:  ({stats['mean']:,.2f}) Promedio de los valores.")
        self.log(f" • std:   ({stats['std']:,.2f}) Desviación estándar (dispersión de los datos).")
        self.log(f" • min:   ({stats['min']:,.2f}) Valor mínimo (el más bajo).")
        self.log(f" • 25%:   ({stats['25%']:,.2f}) Primer cuartil (el 25% de los datos es menor a esto).")
        self.log(f" • 50%:   ({stats['50%']:,.2f}) Mediana (el valor central de los datos).")
        self.log(f" • 75%:   ({stats['75%']:,.2f}) Tercer cuartil (el 75% de los datos es menor a esto).")
        self.log(f" • max:   ({stats['max']:,.2f}) Valor máximo (el más alto).")
        self.log("-" * 60)
        self.log(str(self.df.describe()))
        self.notebook.select(self.tab_data)

    def limpiar_datos(self):
        """
        Paso 2: Limpieza de Datos (Básica).
        Elimina registros inválidos según reglas de negocio:
        1. Valores nulos (NaN).
        2. Valores ilógicos (Superficie <= 10 o Valor <= 0).
        3. Outliers extremos de superficie (Superficie >= 2000).
        """
        if self.df is None:
            messagebox.showerror("Error", "Primero debe cargar los datos.")
            return

        self.log("\n--- 2. LIMPIEZA DE DATOS ---")
        conteo_inicial = self.df.shape[0]
        
        self.df_clean = self.df.dropna().copy()
        nulos_eliminados = conteo_inicial - self.df_clean.shape[0]
        
        temp_len = self.df_clean.shape[0]
        self.df_clean = self.df_clean[(self.df_clean['Superficie'] > 10) & (self.df_clean['Valor'] > 0)]
        ceros_eliminados = temp_len - self.df_clean.shape[0]
        
        # Filtrar por Superficie (Outliers manuales > 2000)
        temp_len = self.df_clean.shape[0]
        self.df_clean = self.df_clean[self.df_clean['Superficie'] < 2000]
        outliers_sup_eliminados = temp_len - self.df_clean.shape[0]

        # Filtrar por Precio (Outliers manuales > 100,000,000 - Errores obvios de digitación)
        temp_len = self.df_clean.shape[0]
        self.df_clean = self.df_clean[self.df_clean['Valor'] < 100000000]
        outliers_val_manual_eliminados = temp_len - self.df_clean.shape[0]
        
        conteo_final = self.df_clean.shape[0]
        total_eliminados = conteo_inicial - conteo_final
        
        self.log(f"Registros originales: {conteo_inicial}")
        self.log(f"Eliminados por nulos: {nulos_eliminados}")
        self.log(f"Eliminados por valores ilógicos: {ceros_eliminados}")
        self.log(f"Eliminados por Superficie excesiva (>2000): {outliers_sup_eliminados}")
        self.log(f"Eliminados por Precio absurdo (>100M): {outliers_val_manual_eliminados}")
        self.log(f"Total eliminados: {total_eliminados}")
        self.log(f"Registros finales: {conteo_final}")
        
        self.update_treeview(self.df_clean)
        self.log("✅ Datos limpios actualizados en la tabla.")

    def regresion_lineal(self):
        """
        Paso 3: Regresión Lineal.
        Entrena un modelo de regresión lineal simple (y = mx + b)
        usando 'Superficie' como variable independiente y 'Valor' como dependiente.
        Muestra los coeficientes y grafica la recta de regresión.
        """
        if self.df_clean is None:
            messagebox.showerror("Error", "Primero debe limpiar los datos.")
            return

        self.log("\n--- 3. REGRESIÓN LINEAL ---")
        X = self.df_clean[['Superficie']]
        y = self.df_clean['Valor']
        
        self.modelo = LinearRegression()
        self.modelo.fit(X, y)
        
        coef = self.modelo.coef_[0]
        intercepto = self.modelo.intercept_
        
        self.log(f"Coeficiente: {coef:.2f}")
        self.log(f"Intercepto: {intercepto:.2f}")
        self.log(f"Ecuación: y = {coef:.2f} * x + {intercepto:.2f}")
        
        self.clear_graph()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(X, y, color='blue', label='Datos Reales')
        ax.plot(X, self.modelo.predict(X), color='red', label='Regresión Lineal')
        ax.set_title('Regresión Lineal: Superficie vs Valor')
        ax.set_xlabel('Superficie (m²)')
        ax.set_ylabel('Valor')
        ax.legend()
        ax.grid(True)
        
        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.notebook.select(self.tab_graphs)

    def pronostico_dialog(self):
        """
        Paso 4: Pronóstico.
        Abre una ventana emergente para que el usuario ingrese superficies.
        Usa el modelo entrenado para predecir el valor de esas superficies.
        """
        if self.modelo is None:
            messagebox.showerror("Error", "Primero debe generar el modelo de regresión.")
            return
            
        # Diálogo simple
        input_window = tk.Toplevel(self.root)
        input_window.title("Pronóstico")
        input_window.geometry("300x150")
        
        ttk.Label(input_window, text="Ingrese superficies (separadas por coma):").pack(pady=10)
        entry = ttk.Entry(input_window)
        entry.pack(pady=5, padx=10, fill=tk.X)
        entry.insert(0, "100, 200, 500")
        
        def calcular():
            try:
                texto = entry.get()
                valores = [float(x.strip()) for x in texto.split(',')]
                df_pred = pd.DataFrame({'Superficie': valores})
                predicciones = self.modelo.predict(df_pred)
                
                self.log("\n--- 4. PRONÓSTICO ---")
                for sup, val in zip(valores, predicciones):
                    self.log(f"Superficie: {sup} m² -> Estimado: ${val:,.2f}")
                
                # --- Actualizar gráfico con los nuevos puntos ---
                self.clear_graph()
                fig, ax = plt.subplots(figsize=(6, 4))
                
                # 1. Graficar datos originales y línea de regresión (Fondo)
                X = self.df_clean[['Superficie']]
                y = self.df_clean['Valor']
                ax.scatter(X, y, color='blue', label='Datos Reales', alpha=0.5)
                ax.plot(X, self.modelo.predict(X), color='red', label='Regresión Lineal')
                
                # 2. Graficar puntos pronosticados (Destacados con X verde grande)
                ax.scatter(valores, predicciones, color='green', s=200, marker='X', label='Pronóstico (Nuevos)', zorder=5)
                
                ax.set_title('Regresión Lineal + Pronóstico')
                ax.set_xlabel('Superficie (m²)')
                ax.set_ylabel('Valor')
                ax.legend()
                ax.grid(True)
                
                canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                self.notebook.select(self.tab_graphs)
                
                input_window.destroy()
            except ValueError:
                messagebox.showerror("Error", "Entrada inválida. Use números separados por comas.")

        ttk.Button(input_window, text="Calcular", command=calcular).pack(pady=10)

    def agrupamiento(self):
        """
        Paso 5: Agrupamiento (Clustering).
        Utiliza el algoritmo K-Means para agrupar las propiedades en 3 clusters
        basados en su Superficie y Valor.
        Visualiza los grupos y sus centroides en un gráfico de dispersión.
        """
        if self.df_clean is None:
            messagebox.showerror("Error", "Primero debe limpiar los datos.")
            return

        self.log("\n--- 5. AGRUPAMIENTO (K-MEANS) ---")
        datos_cluster = self.df_clean[['Superficie', 'Valor']]
        
        self.kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        self.kmeans.fit(datos_cluster)
        
        self.df_clean['Grupo'] = self.kmeans.labels_
        centroids = self.kmeans.cluster_centers_
        
        self.log("Centroides (Superficie, Valor):")
        for i, c in enumerate(centroids):
            self.log(f"Grupo {i}: {c[0]:.2f}, ${c[1]:.2f}")
            
        self.update_treeview(self.df_clean)
        
        self.clear_graph()
        fig, ax = plt.subplots(figsize=(6, 4))
        scatter = ax.scatter(self.df_clean['Superficie'], self.df_clean['Valor'], c=self.df_clean['Grupo'], cmap='viridis')
        ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X', label='Centroides')
        ax.set_title('Agrupamiento K-Means')
        ax.set_xlabel('Superficie')
        ax.set_ylabel('Valor')
        fig.colorbar(scatter, ax=ax, label='Grupo')
        ax.legend()
        ax.grid(True)
        
        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.notebook.select(self.tab_graphs)

if __name__ == "__main__":
    root = tk.Tk()
    app = SistemaMineriaGUI(root)
    root.mainloop()