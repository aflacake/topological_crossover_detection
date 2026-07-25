import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, binary_dilation
from scipy.stats import linregress

def compute_cluster_scaling(grid, p_c=0.3116, min_v=50):
    """
    Mengekstrak eksponen scaling E* khusus untuk klaster dalam scaling regime (V >= min_v).
    """
    struct_6 = np.zeros((3, 3, 3), dtype=int)
    struct_6[1, 1, :] = 1; struct_6[1, :, 1] = 1; struct_6[:, 1, 1] = 1

    labeled, num_feat = label(grid, structure=struct_6)
    if num_feat == 0:
        return None, None

    volumes, areas = [], []
    for cid in range(1, num_feat + 1):
        mask = (labeled == cid)
        v = np.sum(mask)
        if v >= min_v:
            # Hitung permukaan eksak (boundary voxel)
            dilated = binary_dilation(mask)
            a = np.sum(dilated & (~mask))
            volumes.append(v)
            areas.append(a)

    if len(volumes) < 10:
        return None, None

    log_v = np.log10(volumes)
    log_a = np.log10(areas)
    
    slope, intercept, r_val, _, std_err = linregress(log_v, log_a)
    return slope, std_err

def run_irreducible_exponent_experiment(L_list=[24, 32, 40, 48], realizations_per_L=6):
    print("==================================================")
    print("   UJI SKENARIO 2: EKSTRAPOLASI ASIMPTOTIK E*     ")
    print("==================================================")

    p_c = 0.3116
    e_star_means = []
    e_star_errs = []
    inv_L = [1.0 / L for L in L_list]

    for L in L_list:
        slopes = []
        print(f"Memproses L = {L}...", end="")
        for r in range(realizations_per_L):
            grid = (np.random.rand(L, L, L) < p_c).astype(int)
            slope, err = compute_cluster_scaling(grid, p_c=p_c, min_v=40)
            if slope is not None:
                slopes.append(slope)
        
        mean_s = np.mean(slopes)
        std_s = np.std(slopes) / np.sqrt(len(slopes))
        e_star_means.append(mean_s)
        e_star_errs.append(std_s)
        print(f" -> E*(L={L}) = {mean_s:.4f} ± {std_s:.4f}")

    # Ekstrapolasi Linier L -> infinity (1/L -> 0)
    slope_fss, intercept_fss, r_fss, _, err_fss = linregress(inv_L, e_star_means)
    e_star_inf = intercept_fss  # Nilai pada 1/L = 0

    print("\n--------------------------------------------------")
    print("HASIL EKSTRAPOLASI FINITE-SIZE SCALING (L -> ∞):")
    print("--------------------------------------------------")
    print(f"Eksponen Kritis Asimptotik (E* ∞) : {e_star_inf:.4f} ± {err_fss:.4f}")
    print(f"Nilai Pembanding Classical Ds/df : ~0.8720")
    print(f"Selisih Definitif (Deviation)     : {abs(e_star_inf - 0.872):.4f}")
    print("--------------------------------------------------")

    # Visualisasi Finite-Size Scaling Plot
    plt.figure(figsize=(8, 6))
    plt.errorbar(inv_L, e_star_means, yerr=e_star_errs, fmt='o-', color='darkblue', 
                 linewidth=2, capsize=5, markersize=7, label=r'Data Simulasi $E^*(L)$')
    
    # Plot Garis Fit Ekstrapolasi
    x_fit = np.linspace(0, max(inv_L) * 1.1, 50)
    y_fit = slope_fss * x_fit + intercept_fss
    plt.plot(x_fit, y_fit, 'r--', label=rf'Ekstrapolasi $L \to \infty$: $E^*_\infty \approx {e_star_inf:.4f}$')

    plt.axhline(0.872, color='purple', linestyle=':', label=r'Rasio Klasik $D_s/d_f \approx 0.872$')
    
    plt.xlabel(r'Invers Ukuran Kisi ($1/L$)', fontsize=11, fontweight='bold')
    plt.ylabel(r'Eksponen Terukur $E^*(L)$', fontsize=11, fontweight='bold')
    plt.title(r'Finite-Size Scaling untuk Menentukan $E^*_\infty$ (Skenario 2)', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_irreducible_exponent_experiment(L_list=[24, 32, 40, 48], realizations_per_L=5)