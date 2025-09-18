import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import os
from Parametros import generar_parametros
from CargaArchivos import cargar_dataset
from Entrenamiento import entrenar_delta  # <-- NUEVO

class PerceptronUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Perceptrón simple - Interfaz')
        self.geometry('1100x700')
        self.X = None
        self.y = None
        self.parametros = None  # <-- NUEVO: Guardaremos los parámetros aquí
        self.create_widgets()

    def create_widgets(self):
        # Panel izquierdo
        left = ttk.Frame(self)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        # === Dataset ===
        ttk.Label(left, text='Dataset').pack(anchor=tk.W)
        ttk.Button(left, text='Cargar dataset', command=self.cargar_dataset_ui).pack(fill=tk.X)
        self.ds_info = tk.Text(left, width=40, height=5)
        self.ds_info.pack()

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        # === Parámetros iniciales ===
        ttk.Label(left, text='Parámetros iniciales').pack(anchor=tk.W)
        frm = ttk.Frame(left)
        frm.pack(fill=tk.X)

        ttk.Label(frm, text='Tasa aprendizaje (η)').grid(row=0, column=0, sticky=tk.W)
        self.eta_entry = ttk.Entry(frm)
        self.eta_entry.grid(row=0, column=1)

        ttk.Label(frm, text='Max iteraciones').grid(row=1, column=0, sticky=tk.W)
        self.max_iter_entry = ttk.Entry(frm)
        self.max_iter_entry.grid(row=1, column=1)

        ttk.Label(frm, text='Error tolerado (ϵ)').grid(row=2, column=0, sticky=tk.W)
        self.error_entry = ttk.Entry(frm)
        self.error_entry.grid(row=2, column=1)

        ttk.Label(frm, text='Umbral / bias').grid(row=3, column=0, sticky=tk.W)
        self.umbral_entry = ttk.Entry(frm)
        self.umbral_entry.grid(row=3, column=1)
        ttk.Label(frm, text='Vacío = aleatorio').grid(row=3, column=2)

        ttk.Label(frm, text='Pesos iniciales').grid(row=4, column=0, sticky=tk.W)
        self.pesos_entry = ttk.Entry(frm)
        self.pesos_entry.grid(row=4, column=1)
        ttk.Label(frm, text='ej: "0.1,0.2,0.3"').grid(row=4, column=2)

        ttk.Button(left, text='Inicializar perceptrón', command=self.inicializar_perceptron).pack(fill=tk.X, pady=4)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        # === Entrenamiento ===
        ttk.Button(left, text='Iniciar entrenamiento', command=self.iniciar_entrenamiento).pack(fill=tk.X)  # <--
        ttk.Button(left, text='Detener entrenamiento', command=self.fake_action).pack(fill=tk.X)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        # === Simulación / pruebas ===
        ttk.Label(left, text='Simulación / Pruebas').pack(anchor=tk.W)
        ttk.Button(left, text='Probar patrón del dataset', command=self.fake_action).pack(fill=tk.X)

        ttk.Label(left, text='Ingresar nuevo patrón (coma sep)').pack(anchor=tk.W)
        ttk.Entry(left).pack(fill=tk.X)
        ttk.Button(left, text='Probar patrón nuevo', command=self.fake_action).pack(fill=tk.X)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        ttk.Button(left, text='Guardar pesos a CSV', command=self.fake_action).pack(fill=tk.X)

        # === Panel derecho ===
        right = ttk.Frame(self)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Gráfica
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title('Evolución del error (RMS)')
        self.ax.set_xlabel('Iteración')
        self.ax.set_ylabel('RMS error')
        self.line, = self.ax.plot([], [])
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Consola
        ttk.Label(right, text='Consola').pack(anchor=tk.W)
        self.log = tk.Text(right, height=10)
        self.log.pack(fill=tk.BOTH, expand=False)

    # ======= Lógica =======
    def cargar_dataset_ui(self):
        ruta = filedialog.askopenfilename(filetypes=[("Datasets", "*.csv *.json *.xls *.xlsx")])
        if not ruta:
            return

        try:
            self.X, self.y = cargar_dataset(ruta)
            nombre = os.path.basename(ruta)
            info = f"Archivo: {nombre}\nEntradas: {self.X.shape[1]}\nSalidas: 1\nPatrones: {self.X.shape[0]}"
            self.ds_info.delete("1.0", tk.END)
            self.ds_info.insert(tk.END, info)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def inicializar_perceptron(self):
        if self.X is None:
            messagebox.showwarning("Advertencia", "Debes cargar un dataset primero.")
            return

        # Leer valores de los entry
        pesos = self.pesos_entry.get().strip()
        pesos = [float(x) for x in pesos.split(',')] if pesos else None

        umbral = self.umbral_entry.get().strip()
        umbral = float(umbral) if umbral else None

        tasa = self.eta_entry.get().strip()
        tasa = float(tasa) if tasa else None

        max_iter = self.max_iter_entry.get().strip()
        max_iter = int(max_iter) if max_iter else None

        error_max = self.error_entry.get().strip()
        error_max = float(error_max) if error_max else None

        # Generar parámetros
        self.parametros = generar_parametros(
            num_entradas=self.X.shape[1],
            pesos=pesos,
            umbral=umbral,
            tasa=tasa,
            max_iter=max_iter,
            error_max=error_max
        )

        # Mostrar parámetros en consola
        self.log.insert(tk.END, f"\nParámetros inicializados:\n{self.parametros}\n")
        self.log.see(tk.END)

    def iniciar_entrenamiento(self):
        if self.X is None or self.parametros is None:
            messagebox.showwarning("Advertencia", "Debes cargar dataset e inicializar primero.")
            return

        # Ejecutar entrenamiento
        pesos, umbral, errores = entrenar_delta(
            self.X, self.y,
            self.parametros["pesos"],
            self.parametros["umbral"],
            self.parametros["tasa"],
            self.parametros["max_iter"],
            self.parametros["error_max"]
        )

        # Mostrar resultados finales
        self.log.insert(tk.END, f"\nEntrenamiento finalizado.\nPesos: {pesos}\nUmbral: {umbral}\n")
        self.log.see(tk.END)

        # Actualizar gráfica con los errores
        self.line.set_data(range(len(errores)), errores)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()

    def fake_action(self):
        messagebox.showinfo("Acción", "Acción no implementada aún.")


if __name__ == '__main__':
    app = PerceptronUI()
    app.mainloop()



























































































# # Perceptron.py
# import os
# import tkinter as tk
# from tkinter import filedialog, messagebox
# from CargaArchivos import cargar_dataset
# from Parametros import generar_parametros
# import numpy as np

# # ------------------------------------
# # Estado global (se actualiza al cargar archivo / params)
# # ------------------------------------
# num_entradas = 0
# num_salidas = 0
# num_patrones = 0
# current_filepath = ""
# current_filename = ""
# parametros = None  # guardará el dict devuelto por generar_parametros
# X = None
# y = None

# # ------------------------------------
# # Función: seleccionar y cargar archivo
# # ------------------------------------
# def seleccionar_archivo():
#     global num_entradas, num_salidas, num_patrones, current_filepath, current_filename, X, y

#     ruta = filedialog.askopenfilename(
#         title="Selecciona un archivo",
#         filetypes=[
#             ("Archivos CSV", "*.csv"),
#             ("Archivos Excel", "*.xlsx *.xls"),
#             ("Archivos JSON", "*.json"),
#             ("Todos los archivos", "*.*")
#         ]
#     )

#     if not ruta:
#         return

#     try:
#         # Cargar datos desde el módulo de carga
#         X, y = cargar_dataset(ruta)

#         # Actualizar contadores
#         num_entradas = X.shape[1]
#         num_salidas = 1 if len(y.shape) == 1 else y.shape[1]
#         num_patrones = X.shape[0]

#         # Guardar ruta y nombre (solo nombre)
#         current_filepath = ruta
#         current_filename = os.path.basename(ruta)

#         # Actualizar labels en la ventana principal (esquina superior izquierda)
#         lbl_archivo.config(text=f"Archivo: {current_filename}")
#         lbl_entradas.config(text=f"Entradas: {num_entradas}")
#         lbl_salidas.config(text=f"Salidas: {num_salidas}")
#         lbl_patrones.config(text=f"Patrones: {num_patrones}")

#         # Habilitar botón de parámetros
#         btn_param.config(state="normal")

#     except Exception as e:
#         messagebox.showerror("Error", f"No se pudo cargar el archivo:\n{e}")

# # ------------------------------------
# # Función: ventana para configurar parámetros
# # ------------------------------------
# def mostrar_parametros():
#     global parametros
#     if num_entradas == 0:
#         messagebox.showwarning("Atención", "Primero carga un dataset para conocer el número de entradas.")
#         return

#     ventana_param = tk.Toplevel(ventana)
#     ventana_param.title("Configurar parámetros")
#     ventana_param.geometry("480x{}+200+100".format(120 + 30 * num_entradas))

#     # Indicaciones
#     tk.Label(ventana_param, text="Pesos iniciales (dejar vacío para aleatorios)", anchor="w").grid(row=0, column=0, columnspan=3, pady=6, sticky="w")

#     # Entradas para pesos (uno por cada entrada)
#     peso_entries = []
#     for i in range(num_entradas):
#         tk.Label(ventana_param, text=f"w{i+1}").grid(row=1 + i, column=0, padx=6, pady=2, sticky="e")
#         e = tk.Entry(ventana_param, width=12)
#         e.grid(row=1 + i, column=1, padx=6, pady=2, sticky="w")
#         peso_entries.append(e)

#     fila_base = 1 + num_entradas

#     # Campo umbral
#     tk.Label(ventana_param, text="Umbral (θ)").grid(row=fila_base, column=0, padx=6, pady=6, sticky="e")
#     entry_umbral = tk.Entry(ventana_param, width=12)
#     entry_umbral.grid(row=fila_base, column=1, padx=6, pady=6, sticky="w")

#     # Botón para generar aleatorios (pesos + umbral)
#     def rellenar_aleatorios():
#         w_ale = np.random.uniform(-1, 1, size=num_entradas)
#         for i, e in enumerate(peso_entries):
#             e.delete(0, tk.END)
#             e.insert(0, f"{w_ale[i]:.4f}")
#         entry_umbral.delete(0, tk.END)
#         entry_umbral.insert(0, f"{np.random.uniform(-1,1):.4f}")

#     tk.Button(ventana_param, text="Generar aleatorios", command=rellenar_aleatorios).grid(row=fila_base, column=2, padx=8)

#     # Tasa de aprendizaje η
#     tk.Label(ventana_param, text="Tasa aprendizaje (η) [0-1]").grid(row=fila_base+1, column=0, padx=6, pady=2, sticky="e")
#     entry_tasa = tk.Entry(ventana_param, width=12)
#     entry_tasa.grid(row=fila_base+1, column=1, padx=6, pady=2, sticky="w")

#     # Máx iteraciones
#     tk.Label(ventana_param, text="Máx iteraciones").grid(row=fila_base+2, column=0, padx=6, pady=2, sticky="e")
#     entry_max_iter = tk.Entry(ventana_param, width=12)
#     entry_max_iter.grid(row=fila_base+2, column=1, padx=6, pady=2, sticky="w")

#     # Error máximo permitido ϵ
#     tk.Label(ventana_param, text="Error máximo (ϵ)").grid(row=fila_base+3, column=0, padx=6, pady=2, sticky="e")
#     entry_error_max = tk.Entry(ventana_param, width=12)
#     entry_error_max.grid(row=fila_base+3, column=1, padx=6, pady=2, sticky="w")

#     # Función para leer y validar entradas y crear parámetros
#     def crear_parametros():
#         try:
#             # Leer pesos ingresados (solo aquellos que no estén vacíos)
#             pesos_usuario = []
#             for e in peso_entries:
#                 val = e.get().strip()
#                 if val != "":
#                     pesos_usuario.append(float(val))

#             # Si el usuario ingresó menos pesos de los requeridos, lo interpretamos como "usar aleatorios"
#             pesos_final = None
#             if len(pesos_usuario) == num_entradas:
#                 pesos_final = pesos_usuario
#             elif len(pesos_usuario) > 0 and len(pesos_usuario) != num_entradas:
#                 messagebox.showwarning("Advertencia", "Si ingresas pesos debes completar todos los w1..wn. De lo contrario déjalos vacíos para generar aleatorios.")
#                 return

#             # Umbral: si ingresado se usa; si vacío, None para generar aleatorio más adelante
#             umbral_val = entry_umbral.get().strip()
#             umbral_final = float(umbral_val) if umbral_val != "" else None

#             # Tasa de aprendizaje: validar entre 0 y 1
#             tasa_val = entry_tasa.get().strip()
#             tasa_final = float(tasa_val) if tasa_val != "" else 0.1
#             if not (0 <= tasa_final <= 1):
#                 messagebox.showerror("Error", "La tasa de aprendizaje debe estar entre 0 y 1.")
#                 return

#             # Max iteraciones
#             max_iter_val = entry_max_iter.get().strip()
#             max_iter_final = int(max_iter_val) if max_iter_val != "" else 100
#             if max_iter_final <= 0:
#                 messagebox.showerror("Error", "Máx iteraciones debe ser entero positivo.")
#                 return

#             # Error máximo permitido
#             error_val = entry_error_max.get().strip()
#             error_final = float(error_val) if error_val != "" else 0.01
#             if error_final < 0:
#                 messagebox.showerror("Error", "Error máximo debe ser >= 0.")
#                 return

#             # Llamar a la función que construye/normaliza parametros
#             parametros = generar_parametros(
#                 num_entradas=num_entradas,
#                 pesos=pesos_final,
#                 umbral=umbral_final,
#                 tasa=tasa_final,
#                 max_iter=max_iter_final,
#                 error_max=error_final
#             )

#             # Actualizar resumen de parámetros en la ventana principal
#             resumen = (
#                 f"w: [{', '.join(f'{v:.4f}' for v in parametros['pesos'])}]\n"
#                 f"θ: {parametros['umbral']:.4f}  η: {parametros['tasa']:.3f}  "
#                 f"max_iter: {parametros['max_iter']}  ϵ: {parametros['error_max']}"
#             )
#             lbl_parametros.config(text=resumen)

#             # Guardar parámetros globalmente
#             ventana_param.parametros_creados = parametros  # atributo opcional
#             ventana_param.destroy()

#         except ValueError:
#             messagebox.showerror("Error", "Por favor ingresa valores numéricos válidos en los campos.")

#     # Botón aceptar
#     tk.Button(ventana_param, text="Aceptar", command=crear_parametros).grid(row=fila_base+4, column=0, columnspan=3, pady=10)

# # ------------------------------------
# # GUI principal (posicionado en esquina superior izquierda)
# # ------------------------------------
# ventana = tk.Tk()
# ventana.title("Perceptrón - Cargar dataset")
# ventana.geometry("1024x700")

# # Botón: seleccionar archivo (esquina superior izquierda)
# btn_select = tk.Button(ventana, text="Seleccionar archivo", command=seleccionar_archivo)
# btn_select.place(x=10, y=10)

# # Botón: configurar parámetros (deshabilitado hasta cargar dataset)
# btn_param = tk.Button(ventana, text="Configurar parámetros", command=mostrar_parametros, state="disabled")
# btn_param.place(x=160, y=10)

# # Labels en la esquina superior izquierda (información del dataset)
# lbl_archivo = tk.Label(ventana, text="Archivo: (ninguno)", anchor="w", justify="left")
# lbl_archivo.place(x=10, y=50)

# lbl_entradas = tk.Label(ventana, text="Entradas: -", anchor="w")
# lbl_entradas.place(x=10, y=75)

# lbl_salidas = tk.Label(ventana, text="Salidas: -", anchor="w")
# lbl_salidas.place(x=10, y=100)

# lbl_patrones = tk.Label(ventana, text="Patrones: -", anchor="w")
# lbl_patrones.place(x=10, y=125)

# # Label para mostrar resumen de parámetros seleccionados
# lbl_parametros = tk.Label(ventana, text="Parámetros: (no configurados)", anchor="w", justify="left")
# lbl_parametros.place(x=10, y=160)

# ventana.mainloop()
