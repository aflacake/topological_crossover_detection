import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, generate_binary_structure
from scipy.stats import linregress

def check_percolation_spanning(binary_grid):
    """
    Mendeteksi apakah terdapat kluster yang terhubung dari ujung ke ujung 
    (spanning cluster) pada kisi 3D (Z-axis / face-to-face).
    """
    # Menggunakan konektivitas 6-tetangga (muka)
    s = generate_binary_structure(3, 1)
    labeled_grid, num_features = label(binary_grid, structure=s)
    
    if num_features == 0:
        return False, 0
    
    # Cari ID kluster yang ada di permukaan atas (z=0) dan permukaan bawah (z=-1)
    top_labels = np.unique(labeled_grid[0, :, :])
    bottom_labels = np.unique(labeled_grid[-1, :, :])
    
    # Hapus label background (0)
    top_labels = top_labels[top_labels != 0]
    bottom_labels = bottom_labels[bottom_labels != 0]
    
    # Cari irisan (kluster yang menyentuh kedua batas)
    spanning_labels = np.intersect1d(top_labels, bottom_labels)
    
    is_percolating = len(spanning_labels) > 0
    return is_percolating, labeled_grid

def compute_geometry_at_pc(binary_grid):
    """
    Menghitung Volume (V) dan Luas Permukaan Batas (A) untuk kisi 3D.
    """
    V = np.sum(binary_grid)
    if V == 0:
        return 0, 0
    
    # Luas antarmuka terbuka (boundary area) dengan kondisi batas periodik
    diff_x = np.abs(np.diff(np.pad(binary_grid, ((1,1),(0,0),(0,0)), mode='wrap'), axis=0))
    diff_y = np.abs(np.diff(np.pad(binary_grid, ((0,0),(1,1),(0,0)), mode='wrap'), axis=1))
    diff_z = np.abs(np.diff(np.pad(binary_grid, ((0,0),(0,0),(1,1)), mode='wrap'), axis=2))
    A = np.sum(diff_x) + np.sum(diff_y) + np.sum(diff_z)
    
    return V, A

def run_jalur2_fss(L_list=[32, 64, 128], num_samples=3, num_steps=100):
    r"""
    Uji Finite-Size Scaling K_trans tepat pada ambang perkolasi (p_c).
    """
    print("=== MEMULAI SIMULASI JALUR 2: K_trans PADA AMBANG PERKOLASI (p_c) ===")
    
    results = {}
    
    for L in L_list:
        print(f"\n[+] Memproses Skala Kisi L = {L} ...")
        k_values_sample = []
        pc_values_sample = []
        
        for sample in range(num_samples):
            grid_prob = np.random.rand(L, L, L)
            p_values = np.linspace(0.20, 0.40, num_steps)
            
            volumes = []
            areas = []
            p_recorded = []
            spanning_step_idx = -1
            
            for idx, p in enumerate(p_values):
                binary_grid = grid_prob <= p
                V, A = compute_geometry_at_pc(binary_grid)
                
                if V > 0 and A > 0:
                    volumes.append(V)
                    areas.append(A)
                    p_recorded.append(p)
                    
                    # Cek perkolasi pertama kali
                    is_spanning, _ = check_percolation_spanning(binary_grid)
                    if is_spanning and spanning_step_idx == -1:
                        spanning_step_idx = len(volumes) - 1
            
            volumes = np.array(volumes)
            areas = np.array(areas)
            
            if len(volumes) > 0 and spanning_step_idx > 0:
                ln_V = np.log(volumes)
                ln_A = np.log(areas)
                K_trans_curve = np.gradient(ln_A, ln_V)
                
                # Nilai K_trans tepat di titik p_c
                k_at_pc = K_trans_curve[spanning_step_idx]
                p_c_val = p_recorded[spanning_step_idx]
                
                k_values_sample.append(k_at_pc)
                pc_values_sample.append(p_c_val)
                
        if len(k_values_sample) > 0:
            mean_k = np.mean(k_values_sample)
            std_k = np.std(k_values_sample)
            mean_pc = np.mean(pc_values_sample)
            results[L] = (mean_k, std_k, mean_pc)
            print(f"    Amb. Perkolasi (p_c) Terukur : {mean_pc:.4f}")
            print(f"    Rata-rata K_trans (L={L})     : {mean_k:.4f} ± {std_k:.4f}")

    # --- Ekstrapolasi FSS ke Limit L -> Tak Hingga (1/L -> 0) ---
    inv_L = [1.0 / L for L in results.keys()]
    K_means = [results[L][0] for L in results.keys()]
    
    slope, intercept, r_value, p_value, std_err = linregress(inv_L, K_means)
    K_thermodynamic_limit = intercept
    
    print("\n================ HASIL AKHIR JALUR 2 (LIMIT L -> ∞) ================")
    print(f"Konstanta K_trans saat L -> ∞ (1/L = 0) pada p_c : {K_thermodynamic_limit:.4f}")
    print(f"R-squared Regresi                                : {r_value**2:.4f}")
    print(f"Prediksi Hipotesis Awal (d_f / 3)                : 0.8400")
    print(f"Deviasi terhadap 0.8400                          : {abs(K_thermodynamic_limit - 0.8400)/0.8400 * 100:.2f}%")

    # Visualisasi
    plt.figure(figsize=(9, 5.5))
    plt.errorbar(inv_L, K_means, yerr=[results[L][1] for L in results.keys()], 
                 fmt='o', color='darkblue', ecolor='gray', capsize=5, label='Data Simulasi p_c (Rata-Rata)')
    
    x_fit = np.linspace(0, max(inv_L)*1.1, 50)
    plt.plot(x_fit, intercept + slope * x_fit, 'r--', label=f'Fit Linear (Limit L->\\infty: {intercept:.4f})')
    
    plt.axhline(y=0.840, color='orange', linestyle='-.', label=r'Prediksi Awal $d_f / 3$ ($0.840$)')
    plt.axhline(y=0.500, color='green', linestyle=':', label=r'Hukum Kuadratik ($0.500$)')
    plt.axhline(y=0.382, color='purple', linestyle='--', label=r'Hasil Jalur 1 ($\chi = 0$: $0.382$)')
    
    plt.xlabel('Kebalikan Ukuran Kisi (1 / L)')
    plt.ylabel(r'$K_{trans}$ Terukur pada $p_c$')
    plt.title('Finite-Size Scaling $K_{trans}$ Tepat pada Ambang Perkolasi Kritis ($p_c$)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_jalur2_fss(L_list=[32, 64, 128], num_samples=3)