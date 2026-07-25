import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, binary_dilation
from scipy.stats import linregress

def extract_cluster_surface_and_volume(labeled_grid, cluster_id):
    """
    Menghitung Volume (V) dan Luas Permukaan (A) dari satu klaster spesifik.
    """
    cluster_mask = (labeled_grid == cluster_id)
    volume = np.sum(cluster_mask)
    
    # Surface/Hull via Dilation
    dilated = binary_dilation(cluster_mask)
    surface_mask = dilated & (~cluster_mask)
    area = np.sum(surface_mask)
    
    return volume, area

def run_scaling_exponent_experiment(L=40, realizations=5):
    print("==================================================")
    print("   UJI HUKUM SKALA DIRECT LOG-LOG: A ~ V^(E*)     ")
    print("==================================================")
    
    p_c = 0.3116
    struct_6 = np.zeros((3, 3, 3), dtype=int)
    struct_6[1, 1, :] = 1; struct_6[1, :, 1] = 1; struct_6[:, 1, 1] = 1

    volumes = []
    areas = []

    for r in range(realizations):
        grid = (np.random.rand(L, L, L) < p_c).astype(int)
        labeled, num_feat = label(grid, structure=struct_6)
        
        for cid in range(1, num_feat + 1):
            v, a = extract_cluster_surface_and_volume(labeled, cid)
            # Filter klaster mikro (v < 15) agar tidak merusak scaling regime
            if v >= 15 and a > 0:
                volumes.append(v)
                areas.append(a)

    volumes = np.array(volumes)
    areas = np.array(areas)

    log_V = np.log10(volumes)
    log_A = np.log10(areas)

    # Regresi Linier Log-Log untuk mendapatkan Eksponen Slope E*
    slope, intercept, r_value, p_value, std_err = linregress(log_V, log_A)

    print(f"Total Klaster Teranalisis (V >= 15) : {len(volumes)}")
    print(f"R-squared (Kualitas Fitting)       : {r_value**2:.4f}")
    print(f"--------------------------------------------------")
    print(f"Eksponen Skala Terukur (Slope E*)   : {slope:.4f} ± {std_err:.4f}")
    print(f"--------------------------------------------------")

    # Visualisasi Direct Scaling Log-Log (Format LaTeX Bersih)
    plt.figure(figsize=(9, 7))
    plt.scatter(log_V, log_A, alpha=0.4, c='crimson', edgecolors='none', s=20, label='Data Klaster ($p_c$)')
    
    fit_x = np.linspace(min(log_V), max(log_V), 100)
    fit_y = slope * fit_x + intercept
    plt.plot(fit_x, fit_y, 'k--', linewidth=2, label=rf'Fit Hukum Skala: $A \sim V^{{{slope:.3f}}}$')

    plt.xlabel(r'$\log_{10}(\text{Volume } V)$', fontsize=12, fontweight='bold')
    plt.ylabel(r'$\log_{10}(\text{Luas Permukaan } A)$', fontsize=12, fontweight='bold')
    plt.title(f'Uji Direct Scaling $A \\sim V^{{E^*}}$ pada $p_c \\approx 0.3116$ ($L={L}$)', fontsize=13, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_scaling_exponent_experiment(L=40, realizations=5)