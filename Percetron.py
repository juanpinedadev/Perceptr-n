import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import os
from Parametros import generar_parametros
from CargaArchivos import cargar_dataset
from Entrenamiento import entrenar_delta


class PerceptronUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Perceptrón simple - Interfaz')
        self.geometry('1100x700')
        self.X = None
        self.y = None
        self.parametros = None
        self.create_widgets()

    def create_widgets(self):
        # Panel izquierdo
        left = ttk.Frame(self)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        # === Dataset ===
        ttk.Label(left, text='Dataset').pack(anchor=tk.W)
        ttk.Button(left, text='Cargar dataset', command=self.cargar_dataset_ui).pack(fill=tk.X)
        self.ds_info = tk.Text(left, width=40, height=5, state=tk.DISABLED)
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
        ttk.Button(left, text='Iniciar entrenamiento', command=self.iniciar_entrenamiento).pack(fill=tk.X)
        ttk.Button(left, text='Detener entrenamiento', command=self.fake_action).pack(fill=tk.X)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        # === Simulación / pruebas ===
        ttk.Label(left, text='Simulación / Pruebas').pack(anchor=tk.W)
        ttk.Button(left, text='Probar patrón del dataset', command=self.probar_dataset).pack(fill=tk.X)

        ttk.Label(left, text='Ingresar nuevo patrón (coma sep)').pack(anchor=tk.W)
        self.nuevo_patron_entry = ttk.Entry(left)
        self.nuevo_patron_entry.pack(fill=tk.X)
        ttk.Button(left, text='Probar patrón nuevo', command=self.probar_patron_nuevo).pack(fill=tk.X)

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
        self.log = tk.Text(right, height=10, state=tk.DISABLED)
        self.log.pack(fill=tk.BOTH, expand=False)
        
        # Configurar tags para colores
        self.log.tag_configure("success", foreground="green")
        self.log.tag_configure("header", foreground="blue", font=("Arial", 10, "bold"))
        self.log.tag_configure("param", foreground="darkblue")
        self.log.tag_configure("warning", foreground="orange")

    # ======= Lógica =======
    def cargar_dataset_ui(self):
        ruta = filedialog.askopenfilename(filetypes=[("Datasets", "*.csv *.json *.xls *.xlsx")])
        if not ruta:
            return

        try:
            self.X, self.y = cargar_dataset(ruta)
            nombre = os.path.basename(ruta)
            info = f"Archivo: {nombre}\nEntradas: {self.X.shape[1]}\nSalidas: 1\nPatrones: {self.X.shape[0]}"
            
            self.ds_info.config(state=tk.NORMAL)
            self.ds_info.delete("1.0", tk.END)
            self.ds_info.insert(tk.END, info)
            self.ds_info.config(state=tk.DISABLED)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def inicializar_perceptron(self):
        if self.X is None:
            messagebox.showwarning("Advertencia", "Debes cargar un dataset primero.")
            return

        pesos = self.pesos_entry.get().strip()
        pesos = [float(x) for x in pesos.split(',')] if pesos else None
        umbral = float(self.umbral_entry.get().strip()) if self.umbral_entry.get().strip() else None
        tasa = float(self.eta_entry.get().strip()) if self.eta_entry.get().strip() else None
        max_iter = int(self.max_iter_entry.get().strip()) if self.max_iter_entry.get().strip() else None
        error_max = float(self.error_entry.get().strip()) if self.error_entry.get().strip() else None

        self.parametros = generar_parametros(
            num_entradas=self.X.shape[1],
            pesos=pesos,
            umbral=umbral,
            tasa=tasa,
            max_iter=max_iter,
            error_max=error_max
        )

        self.log.config(state=tk.NORMAL)
        self.log.insert(tk.END, "\n", "header")
        self.log.insert(tk.END, "=== PARÁMETROS INICIALIZADOS ===\n", "header")
        pesos_str = ", ".join([f"x{i+1}={p:.6f}" for i, p in enumerate(self.parametros["pesos"])])
        self.log.insert(tk.END, f"Pesos: [{pesos_str}]\n", "param")
        self.log.insert(tk.END, f"Umbral: {self.parametros['umbral']:.6f}\n", "param")
        self.log.insert(tk.END, f"Tasa de aprendizaje: {self.parametros['tasa']}\n", "param")
        self.log.insert(tk.END, f"Máximo de iteraciones: {self.parametros['max_iter']}\n", "param")
        self.log.insert(tk.END, f"Error máximo permitido: {self.parametros['error_max']}\n", "param")
        self.log.insert(tk.END, "===============================\n\n", "header")
        self.log.see(tk.END)
        self.log.config(state=tk.DISABLED)

    def iniciar_entrenamiento(self):
        if self.X is None or self.parametros is None:
            messagebox.showwarning("Advertencia", "Debes cargar dataset e inicializar primero.")
            return

        pesos, umbral, errores = entrenar_delta(
            self.X, self.y,
            self.parametros["pesos"],
            self.parametros["umbral"],
            self.parametros["tasa"],
            self.parametros["max_iter"],
            self.parametros["error_max"]
        )

        iteraciones_realizadas = len(errores)
        terminacion = "convergencia" if iteraciones_realizadas < self.parametros["max_iter"] else "límite de iteraciones"

        self.log.config(state=tk.NORMAL)
        self.log.insert(tk.END, "\n", "header")
        self.log.insert(tk.END, "=== ENTRENAMIENTO FINALIZADO ===\n", "header")
        pesos_str = ", ".join([f"x{i+1}={p:.6f}" for i, p in enumerate(pesos)])
        self.log.insert(tk.END, f"Pesos finales: [{pesos_str}]\n", "param")
        self.log.insert(tk.END, f"Umbral final: {umbral:.6f}\n", "param")
        self.log.insert(tk.END, f"Iteraciones realizadas: {iteraciones_realizadas}\n", "param")
        self.log.insert(tk.END, f"Error final: {errores[-1]:.6f}\n", "param")
        self.log.insert(tk.END, f"Terminación por: {terminacion}\n", "success" if terminacion == "convergencia" else "warning")
        self.log.insert(tk.END, "===============================\n\n", "header")
        self.log.see(tk.END)
        self.log.config(state=tk.DISABLED)

        self.line.set_data(range(len(errores)), errores)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()

    # ==================== NUEVO: probar dataset =====================
    def probar_dataset(self):
        if self.X is None or self.parametros is None:
            messagebox.showwarning("Advertencia", "Debes cargar y entrenar primero.")
            return

        top = tk.Toplevel(self)
        top.title("Resultados Dataset")
        txt = tk.Text(top, width=70, height=25)
        txt.pack(fill=tk.BOTH, expand=True)

        txt.tag_configure("ok", foreground="green")
        txt.tag_configure("fail", foreground="red")

        # función escalón
        def activacion(x):
            return 1 if x >= 0 else 0

        for xi, yi in zip(self.X.values, self.y.values):
            net = sum(w*x for w, x in zip(self.parametros["pesos"], xi)) + self.parametros["umbral"]
            salida = activacion(net)
            acierto = salida == yi
            simbolo = "✔" if acierto else "✘"
            tag = "ok" if acierto else "fail"
            txt.insert(tk.END, f"Entrada: {xi.tolist()}, Esperada: {yi}, Calculada: {salida} {simbolo}\n", tag)

    # ==================== NUEVO: probar patrón nuevo =====================
    def probar_patron_nuevo(self):
        if self.parametros is None:
            messagebox.showwarning("Advertencia", "Debes entrenar primero.")
            return

        patron = self.nuevo_patron_entry.get().strip()
        if not patron:
            messagebox.showwarning("Advertencia", "Debes ingresar un patrón.")
            return

        try:
            valores = [float(x) for x in patron.split(",")]
        except:
            messagebox.showerror("Error", "Formato inválido. Usa números separados por comas.")
            return

        def activacion(x):
            return 1 if x >= 0 else 0

        net = sum(w*x for w, x in zip(self.parametros["pesos"], valores)) + self.parametros["umbral"]
        salida = activacion(net)
        messagebox.showinfo("Resultado", f"Patrón: {valores}\nSalida calculada: {salida}")


        

    def fake_action(self):
        messagebox.showinfo("Acción", "Acción no implementada aún.")


if __name__ == '__main__':
    app = PerceptronUI()
    app.mainloop()
