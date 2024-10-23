from numpy import array, zeros, linspace, concatenate
from numpy.linalg import norm
import matplotlib.pyplot as plt


# # Condiciones iniciales para resolver Kepler
# x_0  = 1
# y_0  = 0
# vx_0 = 0
# vy_0 = 1

# # Tiempo de simulacion y numero de intervalos(particiones)
# t_0 = 0
# t_f = 100
# N = 1000

# # Vector de condiciones iniciales de Kepler
# U_0 = array([x_0,y_0,vx_0,vy_0])

# # Vector de instantes temporales
# ####### t = zeros(N)
# t = linspace(t_0,t_f,N)
# # Aunque nuestro paso está ya definido en el linspace necesitamos
# # dt para mas tarde Euler y RungeKutta
# ###### ¿se podría obviar?
# dt = (t_f-t_0)/N

# # Inicializamos matrices de soluciones de los esquemas numericos
# U_euler = zeros([N,len(U_0)])
# U_rk4   = zeros([N,len(U_0)])

# # Inicializamos las funciones que entran a los esquemas(lado derecho)
# F_euler = zeros([N,len(U_0)])

# k1_rk4 = zeros([N,len(U_0)])
# k2_rk4 = zeros([N,len(U_0)])
# k3_rk4 = zeros([N,len(U_0)])
# k4_rk4 = zeros([N,len(U_0)])

def Euler(U, dt, t, F): 

    return U + dt * F(U, t)

def RungeKutta4(U, dt, t, F):
 
    # Calcula los cuatro coeficientes k
    k1 = F(U, t)
    k2 = F(U + dt * k1/2, t + dt/2)
    k3 = F(U + dt * k2/2, t + dt/2)
    k4 = F(U + dt * k3, t + dt)
    
    # Combina los coeficientes para obtener la solución
    return U + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

def Kepler(U, t): 

    x = U[0]; y = U[1]; vx = U[2]; vy = U[3]
    modulo = ( x**2  +y**2 )**1.5

    return  array( [ vx, vy, -x/modulo, -y/modulo ] ) 



# Vector de estado inicial para resolver Kepler
U = array( [ 1, 0, 0, 1 ])

N = 500
# Vectores que guardan los valores 
x =zeros(N) 
y = zeros(N) 
vx = zeros(N) 
vy = zeros(N) 
t = zeros(N) 

# Condiciones iniciales
x[0] = U[0] 
y[0] = U[1]
vx[0] = U[2]
vy = U[3]
t[0] = 0 

for i in range(1, N): 

      dt = 0.1 
      t[i] = dt*i
      U = Euler(U, dt, t, Kepler)
      x[i] = U[0] 
      y[i] = U[1]
    
plt.plot(x, y)
plt.show()

######  Porque en rk tengo q poner range desde 0

for i in range(0, N): 

      dt = 0.1 
      t[i] = dt*i
      U = RungeKutta4(U, dt, t, Kepler)
      x[i] = U[0] 
      y[i] = U[1]
 
plt.plot(x, y)
plt.show()

    
    




























































