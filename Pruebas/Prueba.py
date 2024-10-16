from numpy import array, zeros, linspace, concatenate
from numpy.linalg import norm
import matplotlib.pyplot as plt

# linspace = secuencias de valores espaciados uniformemente
# concatenate = ara unir arrays o matrices, ya sea en forma horizontal o vertical

#################################
#DEFINIMOS LOS PROBLEMAS FÍSICOS
#################################

def Kepler(U, t): # U = vector espacio y velocidad(4 componentes en 2D)
   
   r = U[0:2] # r ocupa las posiciones 1 y 2 de U
   rdot = U[2:4] # rdot ocupa las posiciones 3 y 4 de U
   # F = U'
   F = concatenate( (rdot,-r/norm(r)**3), axis=0)
   
   return F

#####################
#ESQUEMAS NUMÉRICOS
####################

def Euler(F , U, dt, t):
    
    return U + dt * F(U,t)





#####################
#PROBLEMA DE CAUCHY 
#####################
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
    
 def Cauchy(Esquema, F, U_0, t)
    










































































