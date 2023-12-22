import numpy as np

def linear(n, t):
    if type(t) == int:
        return max(1-n*t, 0)
    return np.fmax(1-n*t, np.zeros(len(t)))

def quadratic(n, t):
    if type(t) == int:
        return max(1-n*t**2, 0)
    return np.fmax(1-n*t**2, np.zeros(len(t)))

# Funções auxiliares de diluição
def exponential(alpha, t):
    return np.power(alpha, t)

# Função para descobrir parâmetro
def _rec_dil_param(func, t, rightwise, p0, p1, convergence_counter=1e3):    
    p_mean = (p1 + p0) / 2
    convergence_counter -= 1
    if convergence_counter == 0:
        return p_mean
    has_converged = func(p_mean, t) < 1e-3
    if (has_converged and rightwise) or (not has_converged and not rightwise):        
        return _rec_dil_param(func, t, rightwise, p_mean, p1, convergence_counter)
    else:        
        return _rec_dil_param(func, t, rightwise, p0, p_mean, convergence_counter)        

def get_dilution_parameter(func, t):
    return _rec_dil_param(func, t, func(0, t)<func(1, t), 0, 1)