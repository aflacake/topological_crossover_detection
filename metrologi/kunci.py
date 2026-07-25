import numpy as np

# Data dari Log Investigasi Poin 2 Kamu
L_vals = np.array([20, 32, 48])
inv_L = 1.0 / L_vals
E_means = np.array([0.444630, 0.503337, 0.480762])
E_stds = np.array([0.209941, 0.146087, 0.077730])

# Weight = 1 / sigma^2
weights = 1.0 / (E_stds ** 2)

# Weighted Linear Fit: E*(1/L) = slope * (1/L) + E_infinity
fit_params, cov_matrix = np.polyfit(inv_L, E_means, deg=1, w=np.sqrt(weights), cov=True)

slope = fit_params[0]
E_infinity = fit_params[1]  # Intercept pada 1/L = 0
std_err_E_infinity = np.sqrt(cov_matrix[1, 1])

print("=========================================================")
print("     HASIL PENGLOCKAN TOPOLOGICAL ELASTICITY FIXED POINT ")
print("=========================================================")
print(f"Topological Elasticity Fixed Point (E* infinity) : {E_infinity:.6f}")
print(f"Standard Error Fixed Point                       : ± {std_err_E_infinity:.6f}")
print(f"Kemiringan Penskalaan (Slope)                    : {slope:.6f}")
print("=========================================================")