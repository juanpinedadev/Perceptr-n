# Parametros.py
import numpy as np

def generar_parametros(num_entradas, pesos=None, umbral=None, tasa=None, max_iter=None, error_max=None):
    # Si no se pasan pesos, se generan aleatoriamente
    if pesos is None or len(pesos) == 0:
        pesos = np.random.uniform(-1, 1, size=num_entradas)
    else:
        pesos = np.array(pesos, dtype=float)

    if umbral is None:
        umbral = np.random.uniform(-1, 1)

    if tasa is None:
        tasa = 0.1

    if max_iter is None:
        max_iter = 100

    if error_max is None:
        error_max = 0.01

    return {
        "pesos": pesos,
        "umbral": umbral,
        "tasa": tasa,
        "max_iter": max_iter,
        "error_max": error_max
    }
