import pandas as pd
import os

def cargar_dataset(ruta_archivo):
    # Detectar extensión del archivo
    _, ext = os.path.splitext(ruta_archivo)

    # Leer el archivo según su tipo
    if ext == '.csv':
        df = pd.read_csv(ruta_archivo)
    elif ext == '.json':
        df = pd.read_json(ruta_archivo)
    elif ext in ('.xls', '.xlsx'):
        df = pd.read_excel(ruta_archivo)
    else:
        raise ValueError(f"Formato de archivo no soportado: {ext}")

    # Buscar si existe una columna llamada "salida" (ignorando mayúsculas/minúsculas)
    columnas = df.columns.tolist()
    columnas_lower = [c.lower() for c in columnas]

    if "salida" in columnas_lower:
        # Si existe, usar esa columna como salida
        indice_salida = columnas_lower.index("salida")
        y = df.iloc[:, indice_salida]
        X = df.drop(df.columns[indice_salida], axis=1)
    else:
        # Si no existe, usar la última columna como salida
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]

    return X, y
