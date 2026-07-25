import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def compute_cubical_euler_characteristic(binary_grid):
    n_voxels = np.sum(binary_grid)
    if n_voxels == 0:
        return 0, 0, 0
    
    padded = np.pad(binary_grid, pad_width=1, mode='wrap')
    
    # Faces
    fx = np.sum(padded[:-1, :, :] & padded[1:, :, :])
    fy = np.sum(padded[:, :-1, :] & padded[:, 1:, :])
    fz = np.sum(padded[:, :, :-1] & padded[:, :, 1:])
    n_faces = fx + fy + fz
    
    # Edges
    e_xy = np.sum(padded[:-1, :-1, :] & padded[1:, :-1, :] & padded[:-1, 1:, :] & padded[1:, 1:, :])
    e_xz = np.sum(padded[:-1, :, :-1] & padded[1:, :, :-1] & padded[:-1, :, 1:] & padded[1:, :, 1:])
    e_yz = np.sum(padded[:, :-1, :-1] & padded[:, 1:, :-1] & padded[:, :-1, 1:] & padded[:, 1:, 1:])
    n_edges = e_xy + e_xz + e_yz
    
    # Nodes
    n_nodes = np.sum(
        padded[:-1, :-1, :-1] & padded[1:, :-1, :-1] & padded[:-1, 1:, :-1] & padded[1:, 1:, :-1] &
        padded[:-1, :-1, 1:]  & padded[1:, :-1, 1:]  & padded[:-1, 1:, 1:]  & padded[1:, 1:, 1:]
    )
    
    chi = n_voxels - n_faces + n_edges - n_nodes
    
    # Area
    diff_x = np.abs(np.diff(np.pad(binary_grid, ((1,1),(0,0),(0,0)), mode='wrap'), axis=0))
    diff_y = np.abs(np.diff(np.pad(binary_grid, ((0,0),(1,1),(0,0)), mode='wrap'), axis=1))
    diff_z = np.abs(np.diff(np.pad(binary_grid, ((0,0),(0,0),(1,1)), mode='wrap'), axis=2))
    area = np.sum(diff_x) + np.sum(diff_y) + np.sum(diff_z)
    
    return n_voxels, area, chi

def run_fss_analysis_robust(L_list=[24, 32, 48, 64], num_samples=10, num_steps=80):
    print("=== MEMULAI ANALISIS FSS ROBUST & PRESI TINGGI ===")
    
    results = {}
    
    for L in L_list:
        print(f"\n[+] Memproses Skala Kisi L = {L} ...")
        k_values_sample = []
        
        for sample in range(num_samples):
            grid_prob = np.random.rand(L, L, L)
            # Fokus pada jendela perkolasi kubis 3D (p_c ~ 0.311)
            p_values = np.linspace(0.26, 0.36, num_steps)
            
            volumes, areas, eulers = [], [], []
            
            for p in p_values:
                binary_grid = grid_prob <= p
                V, A, chi = compute_cubical_euler_characteristic(binary_grid)
                if V > 0 and A > 0:
                    volumes.append(V)
                    areas.append(A)
                    eulers.append(chi)
                    
            volumes = np.array(volumes)
            areas = np.array(areas)
            eulers = np.array(eulers)
            
            # Cari persilangan chi = 0 dengan Interpolasi Linier Eksak
            zero_crossings = np.where(np.diff(np.signbit(eulers)))[0]
            
            if len(zero_crossings) > 0:
                # Ambil persilangan tengah/terakhir jika ada fluktuasi lokal
                idx = zero_crossings[len(zero_crossings) // 2]
                
                # Jendela regresi lokal 7 titik di sekitar transisi
                sub_start = max(0, idx - 3)
                sub_end = min(len(volumes), idx + 4)
                
                sub_ln_V = np.log(volumes[sub_start:sub_end])
                sub_ln_A = np.log(areas[sub_start:sub_end])
                
                if len(sub_ln_V) >= 4:
                    slope_k, _, _, _, _ = linregress(sub_ln_V, sub_ln_A)
                    k_values_sample.append(slope_k)
                
        if len(k_values_sample) > 0:
            mean_k = np.mean(k_values_sample)
            std_k = np.std(k_values_sample)
            results[L] = (mean_k, std_k)
            print(f"    Rata-rata K_trans (L={L:2d}) : {mean_k:.4f} ± {std_k:.4f}")
            
    # Regresi Linear FSS
    inv_L = [1.0 / L for L in results.keys()]
    K_means = [results[L][0] for L in results.keys()]
    
    slope, intercept, r_value, p_value, std_err = linregress(inv_L, K_means)
    
    print("\n================ HASIL FSS ROBUST ================")
    print(f"Konstanta K_trans saat L -> tak hingga (1/L = 0) : {intercept:.4f}")
    print(f"Standard Error Intercept                       : {std_err:.4f}")
    print(f"R-squared Regresi                               : {r_value**2:.4f}")

    # Plot Hasil
    plt.figure(figsize=(8, 5))
    plt.errorbar(inv_L, K_means, yerr=[results[L][1] for L in results.keys()], 
                 fmt='o', color='blue', ecolor='gray', capsize=5, label='Data Simulasi (Robust)')
    
    x_fit = np.linspace(0, max(inv_L)*1.1, 50)
    plt.plot(x_fit, intercept + slope * x_fit, 'r--', label=f'Fit Linear (Limit L->\\infty: {intercept:.4f})')
    
    plt.xlabel('Kebalikan Ukuran Kisi (1 / L)')
    plt.ylabel(r'$K_{trans}$ Terukur pada $\chi = 0$')
    plt.title('Ekstrapolasi FSS Robust (Tanpa Bias L=16)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_fss_analysis_robust(L_list=[24, 32, 48, 64], num_samples=10, num_steps=80)