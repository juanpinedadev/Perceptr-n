# Entrenamiento.py
import numpy as np

def funcion_escalon(suma):
    """
    Función de activación escalón:
    Devuelve 1 si la suma >= 0, en caso contrario 0.
    """
    return 1 if suma >= 0 else 0

def entrenar_delta(X, y, pesos, umbral, tasa, max_iter, error_max):
    """
    Entrena un perceptrón usando la regla delta (corrección de error).
    
    Parámetros:
    - X: matriz de entradas (numpy array)
    - y: vector de salidas esperadas
    - pesos: vector de pesos iniciales
    - umbral: valor de umbral (bias)
    - tasa: tasa de aprendizaje (entre 0 y 1)
    - max_iter: número máximo de iteraciones
    - error_max: error máximo permitido

    Devuelve:
    - pesos finales
    - umbral final
    - lista de errores por iteración
    """
    # Asegurar tipos correctos
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    pesos = np.array(pesos, dtype=float)
    n_patrones = X.shape[0]

    errores_por_iteracion = []

    for iteracion in range(max_iter):
        errores = []

        for i in range(n_patrones):
            # Calcular suma ponderada
            suma = np.dot(X[i], pesos) + umbral

            # Salida usando función escalón
            salida = funcion_escalon(suma)

            # Error
            error = y[i] - salida
            errores.append(abs(error))

            # Actualizar pesos y umbral (Regla Delta)
            pesos += tasa * error * X[i]
            umbral += tasa * error

        # Calcular error promedio
        error_prom = np.mean(errores)
        errores_por_iteracion.append(error_prom)

        # Condición de parada por error
        if error_prom <= error_max:
            break

    return pesos, umbral, errores_por_iteracion
