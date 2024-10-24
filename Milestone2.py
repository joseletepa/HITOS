from numpy import array, zeros, linspace, concatenate
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.optimize import newton

# linspace = secuencias de valores espaciados uniformemente
# concatenate = ara unir arrays o matrices, ya sea en forma horizontal o vertical


#Problema físico


def Kepler(U, t): # U = vector espacio y velocidad(4 componentes en 2D)
   
   r = U[0:2] # r ocupa las posiciones 1 y 2 de U
   rdot = U[2:4] # rdot ocupa las posiciones 3 y 4 de U
   # F = U'
   F = concatenate( (rdot,-r/norm(r)**3), axis=0)
   
   return F

#Esquemas numéricos

def Euler(F , U, dt, t):
    
    return U + dt * F(U,t)

def Euler_Implicito(F, U, dt, t): #tambien se le dice Euler inverso

    def G(X):
        return X - U - dt*F(X,t)
    
    return newton(G, U)  # Utiliza como punto inicial el valor de la solución en el instante anterior

def RungeKutta4(F, U, dt, t):
 
    k1 = F(U, t)
    k2 = F(U + dt * k1/2, t + dt/2)
    k3 = F(U + dt * k2/2, t + dt/2)
    k4 = F(U + dt * k3, t + dt)
    
    return U + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)


def Crank_Nicolson(F, U, dt, t):

    def G(X):
        return X - U - dt/2* ( F(U,t) + F(X,t) )
    
    return newton(G, U)



# Vector de estado inicial para resolver Kepler
U_0 = array( [ 1, 0, 0, 1 ])

# Vectores que guardan los valores 
# x = array( zeros(N) )
# y = array( zeros(N) )
# vx = array( zeros(N) )
# vy = array( zeros(N) )
# t = array( zeros(N) )

# Condiciones iniciales
# x[0] = U[0] 
# y[0] = U[1]
# vx[0] = U[2]
# vy = U[3]

t_0 = 0
t_f = 7
N = 500
t = linspace( t_0, t_f, N)

U = zeros([N+1, 4])
U[0,:] = U_0
#Problema de Cauchy
#
# Obtener la solución de un problema dU/dt = F (ccii), dada una CI
# y un esquema temporal
#
# Inputs : 
#          -> Esquema temporal
#          -> Funcion F(U,t)
#          -> Condición inicial
#          -> Partición temporal
#
# Output :  
#          -> Solución para todo "t" y toda componente
#
    
#################################### como hago para que le entren distintos esquemas

def Cauchy(F, Esquema, U_0, t):
    
    N = len(t)-1
    U = zeros((N+1, len(U_0)))

    U[0,:] = U_0
    for n in range(0,N):
        U[n+1,:] = Esquema( F, U[n,:], t[n+1]-t[n], t[n] )

    return U 



U = Cauchy( Kepler, Euler, U_0, t)

plt.plot (U[:, 0],U[:, 1] )
plt.show()

U = Cauchy( Kepler, Euler_Implicito,U_0, t)

plt.plot(U[:, 0],U[:, 1])
plt.show()

U = Cauchy( Kepler, RungeKutta4, U_0, t)

plt.plot(U[:, 0],U[:, 1])
plt.show()

U = Cauchy( Kepler, Crank_Nicolson, U_0, t)

plt.plot(U[:, 0],U[:, 1])
plt.show()







































































