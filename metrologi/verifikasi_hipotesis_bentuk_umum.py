import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd
from scipy.ndimage import label, binary_dilation
from scipy.stats import linregress

def extract_cluster_metrics_and_betti(grid):
    """
    Mengekstrak Volume (V), Luas Permukaan (A), Kerapatan Loop g1 = beta1/V 
    untuk setiap klaster dalam grid.
    """
    struct_6 = np.zeros((3, 3, 3), dtype=int)
    struct_6[1, 1, :] = 1; struct_6[1, :, 1] = 1; struct_6[:, 1, 1] = 1

    labeled, num_feat = label(grid, structure=struct_6)
    clusters_data = []

    for cid in range(1, num_feat + 1):
        mask = (labeled == cid)
        v = np.sum(mask)
        if v < 20: # Filter klaster sangat kecil (efek kisi)
            continue
            
        dilated = binary_dilation(mask)
        a = np.sum(dilated & (~mask))

        # Hitung Persistensi Homologi / Betti 1 (Loop) khusus klaster ini
        # Membuat bounding box lokal agar komputasi GUDHI super cepat
        coords = np.argwhere(mask)
        min_c, max_c = coords.min(axis=0), coords.max(axis=0) + 1
        sub_grid = mask[min_c[0]:max_c[0], min_c[1]:max_c[1], min_c[2]:max_c[2]]

        cc = gd.CubicalComplex(dimensions=sub_grid.shape, top_dimensional_cells=(1 - sub_grid.astype(int)).flatten())
        p_res = cc.persistence()
        
        # Count beta_1 (dimensi 1 persistence)
        beta_1 = sum(1 for dim, (b, d) in p_res if dim == 1 and b <= 0.5)
        g_1 = beta_1 / v  # Kerapatan Loop Topologis

        clusters_data.append({'V': v, 'A': a, 'beta_1': beta_1, 'g_1': g_1})

    return clusters_data

def run_working_hypothesis_test(L=36, realizations=6):
    print("==================================================")
    print("  UJI HIPOTESIS KERJA: E(V) VS KERAPATAN LOOP g1 ")
    print("==================================================")

    p_c = 0.3116
    all_data = []

    for r in range(realizations):
        grid = (np.random.rand(L, L, L) < p_c).astype(int)
        c_data = extract_cluster_metrics_and_betti(grid)
        all_data.extend(c_data)

    # Kelompokkan data berdasarkan bin V untuk mengukur E_eff di tiap rentang g1
    volumes = np.array([d['V'] for d in all_data])
    areas = np.array([d['A'] for d in all_data])
    g1_vals = np.array([d['g_1'] for d in all_data])

    # Binning berdasarkan Kerapatan Loop g1
    bins = np.linspace(0, max(g1_vals)*0.8, 5)
    g1_centers, e_slopes = [], []

    for i in range(len(bins)-1):
        mask = (g1_vals >= bins[i]) & (g1_vals < bins[i+1])
        if np.sum(mask) >= 8:
            log_v = np.log10(volumes[mask])
            log_a = np.log10(areas[mask])
            slope, intercept, r_val, _, _ = linregress(log_v, log_a)
            
            g1_centers.append(np.mean(g1_vals[mask]))
            e_slopes.append(slope)

    # Fit Linier: E = E0 + alpha * g1
    slope_fit, intercept_fit, r_fit, _, std_err = linregress(g1_centers, e_slopes)

    print("\nHASIL FITTING MODEL FENOMENOLOGIS E = E0 + α * g1:")
    print(f"• Intercept (E0)          : {intercept_fit:.4f} (Estimasi saat g1 = 0)")
    print(f"• Koefisien Kopling (α)   : {slope_fit:.4f}")
    print(f"• Koefisien Determinasi R²: {r_fit**2:.4f}")
    print("--------------------------------------------------")

    # Visualisasi Hipotesis Kerja
    plt.figure(figsize=(8, 6))
    plt.plot(g1_centers, e_slopes, 'o', color='darkred', markersize=8, label='Data Eksperimen $(g_1, E)$')
    
    x_line = np.linspace(0, max(g1_centers)*1.1, 50)
    y_line = slope_fit * x_line + intercept_fit
    plt.plot(x_line, y_line, 'b--', label=rf'Fit Hipotesis Kerja: $E(g_1) = {intercept_fit:.3f} + {slope_fit:.3f} g_1$')

    plt.xlabel(r'Kerapatan Loop Topologis $g_1 = \beta_1 / V$', fontsize=11, fontweight='bold')
    plt.ylabel(r'Eksponent Scaling Efektif $E$', fontsize=11, fontweight='bold')
    plt.title(r'Uji Validitas Model $E(g_1) = E_0 + \alpha g_1$', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_working_hypothesis_test(L=36, realizations=5)