import numpy as np
from scipy.stats import linregress

def compute_cubical_euler_characteristic(binary_grid):
    n_voxels = np.sum(binary_grid)
    if n_voxels == 0:
        return 0, 0, 0
    
    padded = np.pad(binary_grid, pad_width=1, mode='wrap')
    
    # Faces (C2)
    fx = np.sum(padded[:-1, :, :] & padded[1:, :, :])
    fy = np.sum(padded[:, :-1, :] & padded[:, 1:, :])
    fz = np.sum(padded[:, :, :-1] & padded[:, :, 1:])
    
    # Edges (C1)
    e_xy = np.sum(padded[:-1, :-1, :] & padded[1:, :-1, :] & padded[:-1, 1:, :] & padded[1:, 1:, :])
    e_xz = np.sum(padded[:-1, :, :-1] & padded[1:, :, :-1] & padded[:-1, :, 1:] & padded[1:, :, 1:])
    e_yz = np.sum(padded[:, :-1, :-1] & padded[:, 1:, :-1] & padded[:, :-1, 1:] & padded[:, 1:, 1:])
    
    # Nodes (C0)
    n_nodes = np.sum(
        padded[:-1, :-1, :-1] & padded[1:, :-1, :-1] & padded[:-1, 1:, :-1] & padded[1:, 1:, :-1] &
        padded[:-1, :-1, 1:]  & padded[1:, :-1, 1:]  & padded[:-1, 1:, 1:]  & padded[1:, 1:, 1:]
    )
    
    chi = n_voxels - (fx + fy + fz) + (e_xy + e_xz + e_yz) - n_nodes
    
    # Surface Area
    diff_x = np.abs(np.diff(np.pad(binary_grid, ((1,1),(0,0),(0,0)), mode='wrap'), axis=0))
    diff_y = np.abs(np.diff(np.pad(binary_grid, ((0,0),(1,1),(0,0)), mode='wrap'), axis=1))
    diff_z = np.abs(np.diff(np.pad(binary_grid, ((0,0),(0,0),(1,1)), mode='wrap'), axis=2))
    area = np.sum(diff_x) + np.sum(diff_y) + np.sum(diff_z)
    
    return n_voxels, area, chi

def run_high_precision_investigation(L_list=[32, 48, 64], num_samples=5, num_steps=500, window_size=7):
    print("=========================================================================")
    print("   UJI HIPOTESIS EKSPONEN UNIVERSAL E* = 1/2 (FIXED & HIGH PRECISION)   ")
    print("=========================================================================\n")
    print(f" * Resolusi Grid p : {num_steps} langkah (p ∈ [0.28, 0.36])")
    print(f" * Estimator       : Fitting Linier Lokal ({window_size} titik di sekitar χ = 0)\n")
    
    results = {}
    
    for L in L_list:
        print(f"[+] Memproses Skala Kisi L = {L:2d} ...")
        e_stars = []
        p_crits = []
        
        for s in range(num_samples):
            np.random.seed(s + 200)
            grid_prob = np.random.rand(L, L, L)
            # Rentang p diperluas [0.28, 0.36] agar p* pasti tertangkap untuk semua L
            p_values = np.linspace(0.28, 0.36, num_steps)
            
            volumes, areas, eulers = [], [], []
            for p in p_values:
                V, A, chi = compute_cubical_euler_characteristic(grid_prob <= p)
                if V > 0 and A > 0:
                    volumes.append(V)
                    areas.append(A)
                    eulers.append(chi)
                    
            volumes = np.array(volumes)
            areas = np.array(areas)
            eulers = np.array(eulers)
            
            zero_crossings = np.where(np.diff(np.signbit(eulers)))[0]
            if len(zero_crossings) == 0:
                print(f"    [WARN] Sample #{s+1} pada L={L} tidak menemukan chi = 0. Dilewati.")
                continue
                
            idx = zero_crossings[len(zero_crossings) // 2]
            
            # Interpolasi p*
            p1, p2 = p_values[idx], p_values[idx+1]
            chi1, chi2 = eulers[idx], eulers[idx+1]
            p_crit = p1 - chi1 * (p2 - p1) / (chi2 - chi1)
            p_crits.append(p_crit)
            
            # Fitting Linier Lokal (window_size titik) di sekitar chi = 0
            half_w = window_size // 2
            sub_start = max(0, idx - half_w)
            sub_end = min(len(volumes), idx + half_w + 1)
            
            sub_ln_V = np.log(volumes[sub_start:sub_end])
            sub_ln_A = np.log(areas[sub_start:sub_end])
            
            slope_E, _, r_value, _, std_err = linregress(sub_ln_V, sub_ln_A)
            
            if not np.isnan(slope_E):
                e_stars.append(slope_E)
                
        if len(e_stars) > 0:
            mean_E = np.mean(e_stars)
            std_E = np.std(e_stars)
            mean_p = np.mean(p_crits)
            results[L] = (mean_E, std_E, mean_p)
            print(f"    -> L = {L:2d} | Mean E* = {mean_E:.6f} | STD (Noise) = {std_E:.6f} | p* = {mean_p:.6f}")
        else:
            print(f"    -> L = {L:2d} | Gagal mengumpulkan data valid.")
            
    print("\n=========================================================================")
    print("           REKAPITULASI PENSKALAAN DENGAN FITTING LOKAL TERHALUS         ")
    print("=========================================================================")
    print(" Ukuran Kisi (L)  |   Rerata E*   |   STD (Noise)   | Selisih dari 0.500000 ")
    print("-------------------------------------------------------------------------")
    
    valid_L = np.array(list(results.keys()))
    inv_L = 1.0 / valid_L
    E_means = np.array([results[L][0] for L in valid_L])
    E_stds = np.array([results[L][1] for L in valid_L])
    
    for L in valid_L:
        m_E, s_E, _ = results[L]
        diff_half = abs(m_E - 0.5)
        print(f"      L = {L:2d}       |   {m_E:.6f}    |    {s_E:.6f}    |      {diff_half:.6f}")
        
    # Ekstrapolasi Weighted Least Squares (WLS) ke L -> inf
    weights = 1.0 / (E_stds ** 2 + 1e-12) # Mencegah pembagian nol
    fit_params, cov_matrix = np.polyfit(inv_L, E_means, deg=1, w=np.sqrt(weights), cov=True)
    
    E_inf_fitted = fit_params[1]
    err_E_inf = np.sqrt(cov_matrix[1, 1])
    
    print("-------------------------------------------------------------------------")
    print(f" Hasil Ekstrapolasi WLS (L -> ∞) E*inf : {E_inf_fitted:.6f} ± {err_E_inf:.6f}")
    print(f" Selisih Mutlak E*inf terhadap 1/2      : {abs(E_inf_fitted - 0.5):.6f}")
    print("=========================================================================\n")

if __name__ == "__main__":
    run_high_precision_investigation(L_list=[32, 48, 64], num_samples=5, num_steps=500, window_size=7)