import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def compute_cubical_geometry_and_euler(binary_grid):
    """
    Menhitung Volume (V), Luas Antarmuka Terbuka (A), dan Karakteristik Euler (chi)
    eksak untuk kisi kubus 3D berdasarkan struktur selular.
    """
    n_voxels = np.sum(binary_grid)
    if n_voxels == 0:
        return 0, 0, 0
    
    padded = np.pad(binary_grid, pad_width=1, mode='wrap')
    
    # Faces (Muka)
    fx = np.sum(padded[:-1, :, :] & padded[1:, :, :])
    fy = np.sum(padded[:, :-1, :] & padded[:, 1:, :])
    fz = np.sum(padded[:, :, :-1] & padded[:, :, 1:])
    n_faces = fx + fy + fz
    
    # Edges (Rusuk)
    e_xy = np.sum(padded[:-1, :-1, :] & padded[1:, :-1, :] & padded[:-1, 1:, :] & padded[1:, 1:, :])
    e_xz = np.sum(padded[:-1, :, :-1] & padded[1:, :, :-1] & padded[:-1, :, 1:] & padded[1:, :, 1:])
    e_yz = np.sum(padded[:, :-1, :-1] & padded[:, 1:, :-1] & padded[:, :-1, 1:] & padded[:, 1:, 1:])
    n_edges = e_xy + e_xz + e_yz
    
    # Nodes (Titik)
    n_nodes = np.sum(
        padded[:-1, :-1, :-1] & padded[1:, :-1, :-1] & padded[:-1, 1:, :-1] & padded[1:, 1:, :-1] &
        padded[:-1, :-1, 1:]  & padded[1:, :-1, 1:]  & padded[:-1, 1:, 1:]  & padded[1:, 1:, 1:]
    )
    
    # Karakteristik Euler: chi = V - F + E - N
    chi = n_voxels - n_faces + n_edges - n_nodes
    
    # Area (Luas Antarmuka Batas Terbuka)
    diff_x = np.abs(np.diff(np.pad(binary_grid, ((1,1),(0,0),(0,0)), mode='wrap'), axis=0))
    diff_y = np.abs(np.diff(np.pad(binary_grid, ((0,0),(1,1),(0,0)), mode='wrap'), axis=1))
    diff_z = np.abs(np.diff(np.pad(binary_grid, ((0,0),(0,0),(1,1)), mode='wrap'), axis=2))
    area = np.sum(diff_x) + np.sum(diff_y) + np.sum(diff_z)
    
    return n_voxels, area, chi

def execute_jalur1_formal_fss(L_list=[32, 64, 128], num_samples=5, num_steps=80):
    r"""
    Eksekusi formal FSS untuk mengukur Kcrit pada titik transisi chi = 0.
    
    Definisi Formal:
    Kcrit = lim_{L -> \infty} [d(ln A) / d(ln V)] |_{\chi = 0}
    """
    print("=" * 65)
    print("      FORMALISME JALUR 1: EKSPLORASI K_crit PADA TRANSISI χ = 0")
    print("=" * 65)
    
    results = {}
    
    for L in L_list:
        k_trans_samples = []
        
        for s in range(num_samples):
            grid_prob = np.random.rand(L, L, L)
            p_values = np.linspace(0.20, 0.45, num_steps)
            
            volumes, areas, eulers = [], [], []
            
            for p in p_values:
                binary_grid = grid_prob <= p
                V, A, chi = compute_cubical_geometry_and_euler(binary_grid)
                if V > 0 and A > 0:
                    volumes.append(V)
                    areas.append(A)
                    eulers.append(chi)
                    
            volumes = np.array(volumes)
            areas = np.array(areas)
            eulers = np.array(eulers)
            
            # Turunan Alometrik Lokal: d(ln A) / d(ln V)
            ln_V = np.log(volumes)
            ln_A = np.log(areas)
            K_curve = np.gradient(ln_A, ln_V)
            
            # Isosolasi kondisi Kelintasan Nol Topologis (chi = 0)
            zero_crossings = np.where(np.diff(np.signbit(eulers)))[0]
            if len(zero_crossings) > 0:
                idx = zero_crossings[0]
                k_val = K_curve[idx]
                k_trans_samples.append(k_val)
                
        if len(k_trans_samples) > 0:
            mean_k = np.mean(k_trans_samples)
            std_k = np.std(k_trans_samples)
            results[L] = (mean_k, std_k)
            print(f"L = {L:3d}  |  K_trans(L) = {mean_k:.4f} ± {std_k:.4f}")

    # --- Finite-Size Scaling (Ekstrapolasi Linear 1/L -> 0) ---
    inv_L = np.array([1.0 / L for L in results.keys()])
    K_means = np.array([results[L][0] for L in results.keys()])
    K_stds = np.array([results[L][1] for L in results.keys()])
    
    slope, intercept, r_value, p_value, std_err = linregress(inv_L, K_means)
    r_squared = r_value**2
    K_crit = intercept
    
    print("-" * 65)
    print(f"Persamaan Trend FSS  : K(L) = {K_crit:.4f} + ({slope:.4f}) * (1/L)")
    print(f"Limit Asimtotik      : K_crit (L -> ∞) = {K_crit:.4f}")
    print(f"Kualitas Fit (R²)    : {r_squared:.4f}")
    print(f"Simpangan Baku L=128 : ±{K_stds[-1]:.4f}")
    print("=" * 65)
    
    # --- Visualisasi Grafik FSS ---
    plt.figure(figsize=(8, 5))
    plt.errorbar(inv_L, K_means, yerr=K_stds, fmt='ob', ecolor='gray', capsize=5, 
                 label='Data Simulasi (Rata-rata ± Std)')
    
    x_fit = np.linspace(0, max(inv_L) * 1.15, 50)
    plt.plot(x_fit, intercept + slope * x_fit, 'r--', 
             label=f'Linear Fit (K_crit = {K_crit:.4f}, R² = {r_squared:.4f})')
    
    plt.axvline(0, color='black', linewidth=0.8, linestyle=':')
    plt.xlabel(r'Kebalikan Ukuran Kisi $(1 / L)$')
    plt.ylabel(r'Eksponent Alometrik Terukur $\left. \frac{d(\ln A)}{d(\ln V)} \right|_{\chi = 0}$')
    plt.title(r'Finite-Size Scaling: Formulasi Asimtotik $K_{crit}$ pada Transisi $\chi = 0$')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    execute_jalur1_formal_fss()