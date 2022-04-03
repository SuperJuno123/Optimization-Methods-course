import numpy as np

learning_rate = 1e-3
accuracy_epsilon = 1e-9
derivative_epsilon = 1e-9
# hessian_epsilon = np.sqrt(derivative_epsilon)
hessian_epsilon=1e-9
n = 16
coefficient_for_Quasi_Newton = 0.1
gamma_for_Markovitz = 1e-1