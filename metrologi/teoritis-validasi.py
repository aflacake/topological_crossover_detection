import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, binary_dilation
from scipy.optimize import curve_fit
import gudhi as gd
import time

def estimate_fractal_dimension_boxcounting(binary_grid):
    """
    Menghitung dimensi fraktal d_f menggunakan metode Box-Counting (3D) yang robust.
    """
    Z = binary_grid > 0
    if not np.any(Z):
        return 0.0
        
    L = Z.shape[0]
    # Pilih ukuran box yang merupakan pembagi dari L atau gunakan kuadrat berpangkat
    sizes = [s for s in [2, 3, 4, 6, 8, 9, 12] if L % s == 0]
    if len(sizes) < 2:
        sizes = [2, 4, 8] # fallback
        
    counts = []
    
    for size in sizes:
        # Potong grid agar ukurannya tepat habis dibagi 'size'
        n_blocks = L // size
        cut_L = n_blocks * size
        Z_cut = Z[:cut_L, :cut_L, :cut_L]
        
        # Reshape 3D grid menjadi himpunan blok 3D berukuran (size, size, size)
        shape = (n_blocks, size, n_blocks, size, n_blocks, size)
        blocks = Z_cut.reshape(shape)
        
        # Hitung berapa banyak blok yang berisi setidaknya 1 voksel (True)
        count = np.sum(blocks.any(axis=(1, 3, 5)))
        counts.append(count)
        
    if len(counts) < 2 or any(c == 0 for c in counts):
        return 0.0

    # Fit linier: log(N) vs log(1/s)
    log_sizes = np.log(1.0 / np.array(sizes))
    log_counts = np.log(counts)
    
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    return coeffs[0]

def extract_surface_hull(binary_grid):
    """
    Mengekstrak permukaan/hull dari klaster 3D (voksel batas).
    """
    dilated = binary_dilation(binary_grid)
    surface = dilated & (~binary_grid)
    return surface

def run_theory_derivation_experiment(L=32, seed=42):
    print("==================================================")
    print("      MEMBUKTIKAN RUMUS TEORI E* DENGAN PYTHON    ")
    print("==================================================")
    
    np.random.seed(seed)
    p_c = 0.3116  # Critical probability literatur 3D
    
    # 1. Generate Sistem pada p_c
    grid = (np.random.rand(L, L, L) < p_c).astype(int)
    
    # 2. Ambil Klaster Terbesar (Spanning/Giant Cluster)
    struct_6 = np.zeros((3, 3, 3), dtype=int)
    struct_6[1, 1, :] = 1; struct_6[1, :, 1] = 1; struct_6[:, 1, 1] = 1
    
    labeled, num_feat = label(grid, structure=struct_6)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    giant_label = sizes.argmax()
    
    giant_cluster = (labeled == giant_label).astype(int)
    giant_surface = extract_surface_hull(giant_cluster)
    
    print(f"Volume Klaster Terbesar (N_voxels) : {np.sum(giant_cluster)}")
    print(f"Luas Permukaan (N_surface_voxels) : {np.sum(giant_surface)}")
    
    # 3. Hitung Dimensi Fraktal d_f (Volume) dan D_s (Permukaan)
    d_f = estimate_fractal_dimension_boxcounting(giant_cluster)
    D_s = estimate_fractal_dimension_boxcounting(giant_surface)
    
    print(f"\nHasil Box-Counting:")
    print(f"• Dimensi Fraktal Volume (d_f)    : {d_f:.4f} (Teori 3D: ~2.52)")
    print(f"• Dimensi Fraktal Permukaan (D_s) : {D_s:.4f} (Teori 3D: ~2.18)")
    
    # 4. Evaluasi Hipotesis Teoritis E*
    pred_hyp_A = D_s / d_f if d_f > 0 else 0
    pred_hyp_B = (d_f - 1) / d_f if d_f > 0 else 0
    
    # 5. Hitung Betti Numbers pada Klaster Terbesar
    cc = gd.CubicalComplex(dimensions=[L, L, L], top_dimensional_cells=(1 - giant_cluster).flatten())
    p_res = cc.persistence()
    
    b0 = sum(1 for dim, (b, d) in p_res if dim == 0 and b <= 0.5)
    b1 = sum(1 for dim, (b, d) in p_res if dim == 1 and b <= 0.5)
    b2 = sum(1 for dim, (b, d) in p_res if dim == 2 and b <= 0.5)
    chi = b0 - b1 + b2
    
    # Rasio Betti Topologis (Hipotesis C)
    # E_betti = 1 - (beta_1 / (beta_0 + beta_1 + beta_2))
    betti_sum = b0 + b1 + b2
    pred_hyp_C = 1.0 - (b1 / betti_sum) if betti_sum > 0 else 0
    
    print("\n--------------------------------------------------")
    print("UJI HIPOTESIS RUMUS TEORI E*:")
    print("--------------------------------------------------")
    print(f"1. Hipotesis A (Ds / df)         : {pred_hyp_A:.4f}")
    print(f"2. Hipotesis B ((df - 1) / df)   : {pred_hyp_B:.4f}")
    print(f"3. Hipotesis C (Topologi Betti) : {pred_hyp_C:.4f}")
    print("--------------------------------------------------")
    
    # Visualisasi Komparasi Model Teori
    fig, ax = plt.subplots(figsize=(8, 6))
    
    models = ['Hipotesis A\n(Ds / df)', 'Hipotesis B\n((df-1) / df)', 'Hipotesis C\n(Betti Ratio)', 'Target Numerik\nE* (~0.872)']
    values = [pred_hyp_A, pred_hyp_B, pred_hyp_C, 0.872]
    colors = ['crimson', 'gray', 'darkblue', 'purple']
    
    bars = ax.bar(models, values, color=colors, width=0.5, alpha=0.85)
    ax.axhline(0.872, color='purple', linestyle='--', label='Nilai E* Empiris (~0.872)')
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom', fontweight='bold')
        
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Nilai Prediksi $E^*$', fontsize=12)
    ax.set_title('Uji Validasi Formulasi Teoritis $E^*$', fontsize=14, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_theory_derivation_experiment(L=36)