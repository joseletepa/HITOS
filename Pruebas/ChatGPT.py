import numpy as np
import matplotlib.pyplot as plt

def metodo_euler(f, y0, t0, tf, h):
    """
    Método de Euler genérico para resolver EDOs de la forma dy/dt = f(t, y)
    
    Parámetros:
    f  -- Función que define la EDO, f(t, y)
    y0 -- Valor inicial de la variable dependiente (en t0)
    t0 -- Tiempo inicial
    tf -- Tiempo final
    h  -- Tamaño del paso
    
    Retorna:
    t_values -- Array de tiempos en los que se evaluó la solución
    y_values -- Aproximación de la solución en cada tiempo
    """
    t_values = np.arange(t0, tf + h, h)  # Vector de tiempos
    y_values = np.zeros(len(t_values))   # Vector para almacenar las soluciones
    y_values[0] = y0                     # Valor inicial

    for i in range(1, len(t_values)):
        y_values[i] = y_values[i-1] + h * f(t_values[i-1], y_values[i-1])  # Fórmula de Euler

    return t_values, y_values

# Ejemplo de uso
if __name__ == "__main__":
    # Definimos la ecuación diferencial dy/dt = -2y (decadencia exponencial)
    def f(t, y):
        return -2 * y

    # Parámetros del problema
    y0 = 1     # Valor inicial de y
    t0 = 0     # Tiempo inicial
    tf = 5     # Tiempo final
    h = 0.1    # Tamaño del paso

    # Llamamos al método de Euler
    t_values, y_values = metodo_euler(f, y0, t0, tf, h)

    # Mostramos los resultados
    plt.plot(t_values, y_values, label="Aproximación por Euler")
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.title('Método de Euler')
    plt.grid(True)
    plt.legend()
    plt.show()
print(t_values)