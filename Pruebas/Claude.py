def metodo_euler(f, y0, t0, tf, h):
    """
    Implementa el método de Euler para resolver una EDO.
    
    Parámetros:
    f  : función que define la EDO dy/dt = f(t, y)
    y0 : valor inicial de y
    t0 : tiempo inicial
    tf : tiempo final
    h  : tamaño del paso
    
    Retorna:
    t : lista de puntos en el tiempo
    y : lista de valores de y correspondientes
    """
    t = [t0]
    y = [y0]
    
    while t[-1] < tf:
        ti = t[-1]
        yi = y[-1]
        
        t_next = ti + h
        y_next = yi + h * f(ti, yi)
        
        t.append(t_next)
        y.append(y_next)
    
    return t, y

# Ejemplo de uso:
def ejemplo_edo(t, y):
    return -2 * y  # EDO: dy/dt = -2y

# Parámetros del problema
y0 = 1  # Condición inicial
t0 = 0  # Tiempo inicial
tf = 2  # Tiempo final
h = 0.1  # Tamaño del paso

# Resolver la EDO
t, y = metodo_euler(ejemplo_edo, y0, t0, tf, h)

# Imprimir resultados
for ti, yi in zip(t, y):
    print(f"t = {ti:.2f}, y = {yi:.6f}")