import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, binary_dilation
from scipy.stats import linregress

def extract_cluster_surface_and_volume(labeled_grid, cluster_id):
    """Mengekstrak Volume (V) dan Outer Boundary Layer (A)."""
    cluster_mask = (labeled_grid == cluster_id)
    volume = np.sum(cluster_mask)
    
    dilated = binary_dilation(cluster_mask)
    surface_mask = dilated & (~cluster_mask)
    area = np.sum(surface_mask)
    
    return volume, area

def run_binned_scaling_analysis(L=48, realizations=10):
    print("==================================================")
    print("    UJI BINNED SCALING: EKSPONEN VS UKURAN KLASTER ")
    print("==================================================")
    
    p_c = 0.3116
    struct_6 = np.zeros((3, 3, 3), dtype=int)
    struct_6[1, 1, :] = 1; struct_6[1, :, 1] = 1; struct_6[:, 1, 1] = 1

    volumes, areas = [], []

    # 1. Kumpulkan Sample Klaster
    for r in range(realizations):
        grid = (np.random.rand(L, L, L) < p_c).astype(int)
        labeled, num_feat = label(grid, structure=struct_6)
        
        for cid in range(1, num_feat + 1):
            v, a = extract_cluster_surface_and_volume(labeled, cid)
            if v >= 15 and a > 0:
                volumes.append(v)
                areas.append(a)

    volumes = np.array(volumes)
    areas = np.array(areas)

    # 2. Bagi Menjadi 3 Kelompok Volume (Regime Binning)
    bin_1 = (volumes >= 15) & (volumes < 50)
    bin_2 = (volumes >= 50) & (volumes < 200)
    bin_3 = (volumes >= 200)

    # 3. Hitung Slope E* Masing-Masing Kelompok
    slope1, _, r1, _, err1 = linregress(np.log10(volumes[bin_1]), np.log10(areas[bin_1])) if np.sum(bin_1) > 5 else (0,0,0,0,0)
    slope2, _, r2, _, err2 = linregress(np.log10(volumes[bin_2]), np.log10(areas[bin_2])) if np.sum(bin_2) > 5 else (0,0,0,0,0)
    slope3, _, r3, _, err3 = linregress(np.log10(volumes[bin_3]), np.log10(areas[bin_3])) if np.sum(bin_3) > 5 else (0,0,0,0,0)

    print("\nHASIL FIT BERDASARKAN KELOMPOK UKURAN (SUB-REGIMES):")
    print(f"1. Rezim Kecil   (15 <= V < 50)  [N={np.sum(bin_1)}]: E* = {slope1:.4f} ± {err1:.4f} (R²={r1**2:.3f})")
    print(f"2. Rezim Menengah(50 <= V < 200) [N={np.sum(bin_2)}]: E* = {slope2:.4f} ± {err2:.4f} (R²={r2**2:.3f})")
    print(f"3. Rezim Besar   (V >= 200)      [N={np.sum(bin_3)}]: E* = {slope3:.4f} ± {err3:.4f} (R²={r3**2:.3f})")
    print("--------------------------------------------------")

    # 4. Visualisasi Binned Fitting Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: Overlay Scatter & Fitting Per Rezim ---
    ax1.scatter(np.log10(volumes[bin_1]), np.log10(areas[bin_1]), color='coral', alpha=0.3, s=15, label=f'Rezim Kecil ($E^*={slope1:.3f}$)')
    ax1.scatter(np.log10(volumes[bin_2]), np.log10(areas[bin_2]), color='teal', alpha=0.5, s=20, label=f'Rezim Menengah ($E^*={slope2:.3f}$)')
    ax1.scatter(np.log10(volumes[bin_3]), np.log10(areas[bin_3]), color='purple', alpha=0.8, s=30, label=f'Rezim Besar ($E^*={slope3:.3f}$)')

    ax1.set_xlabel(r'$\log_{10}(\text{Volume } V)$', fontsize=11, fontweight='bold')
    ax1.set_ylabel(r'$\log_{10}(\text{Luas Permukaan } A)$', fontsize=11, fontweight='bold')
    ax1.set_title(r'Pergeseran Slope $E^*$ Berdasarkan Kelompok Volume', fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend()

    # --- Plot 2: Tren Evolusi E* Terhadap Skala ---
    bins = ['15 ≤ V < 50\n(Kecil)', '50 ≤ V < 200\n(Menengah)', 'V ≥ 200\n(Besar)']
    slopes = [slope1, slope2, slope3]
    errors = [err1, err2, err3]

    ax2.errorbar(bins, slopes, yerr=errors, fmt='o-', color='crimson', linewidth=2, capsize=6, markersize=8)
    ax2.axhline(0.872, color='purple', linestyle='--', label=r'Target Scaling Limit $E^* \approx 0.872$')
    ax2.set_ylabel(r'Eksponent Scaling $E^*$', fontsize=11, fontweight='bold')
    ax2.set_title(r'Evolusi $E^*(V)$ Menuju Limit Asimptotik Makroskopis', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.set_ylim(0.80, 1.00)
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_binned_scaling_analysis(L=48, realizations=8)