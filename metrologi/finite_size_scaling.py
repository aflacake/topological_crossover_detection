import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def compute_euler_exact(grid_3d):
    r"""Menghitung Karakteristik Euler (chi = N0 - N1 + N2 - N3) dari matriks 3D biner."""
    voxels = np.argwhere(grid_3d == 1)
    if len(voxels) == 0:
        return 0
    N3 = len(voxels)
    vertices, edges, faces = set(), set(), set()

    v_offsets = [
        (0,0,0), (1,0,0), (0,1,0), (1,1,0),
        (0,0,1), (1,0,1), (0,1,1), (1,1,1)
    ]
    e_offsets = [
        ((0,0,0),(1,0,0)), ((0,1,0),(1,1,0)), ((0,0,1),(1,0,1)), ((0,1,1),(1,1,1)),
        ((0,0,0),(0,1,0)), ((1,0,0),(1,1,0)), ((0,0,1),(0,1,1)), ((1,0,1),(1,1,1)),
        ((0,0,0),(0,0,1)), ((1,0,0),(1,0,1)), ((0,1,0),(0,1,1)), ((1,1,0),(1,1,1))
    ]
    f_offsets = [
        ((0,0,0),(1,0,0),(1,1,0),(0,1,0)), ((0,0,1),(1,0,1),(1,1,1),(0,1,1)),
        ((0,0,0),(1,0,0),(1,0,1),(0,0,1)), ((0,1,0),(1,1,0),(1,1,1),(0,1,1)),
        ((0,0,0),(0,1,0),(0,1,1),(0,0,1)), ((1,0,0),(1,1,0),(1,1,1),(1,0,1))
    ]

    for vx, vy, vz in voxels:
        for dx, dy, dz in v_offsets:
            vertices.add((vx + dx, vy + dy, vz + dz))
        for (x1,y1,z1), (x2,y2,z2) in e_offsets:
            p1 = (vx + x1, vy + y1, vz + z1)
            p2 = (vx + x2, vy + y2, vz + z2)
            edges.add(tuple(sorted([p1, p2])))
        for f in f_offsets:
            face_verts = tuple(sorted([(vx + dx, vy + dy, vz + dz) for dx, dy, dz in f]))
            faces.add(face_verts)

    return len(vertices) - len(edges) + len(faces) - N3


def measure_area_and_volume(grid_3d):
    r"""Menghitung Volume (V) dan Muka Antarmuka/Area (A) dari kisi perkolasi."""
    voxels = np.argwhere(grid_3d == 1)
    V = len(voxels)
    if V == 0:
        return 0, 0

    # Hitung luas permukaan (A) berdasarkan muka kubus yang terpapar (tidak berdempetan)
    padded = np.pad(grid_3d, 1, mode='constant', constant_values=0)
    diff_x = np.abs(np.diff(padded, axis=0))
    diff_y = np.abs(np.diff(padded, axis=1))
    diff_z = np.abs(np.diff(padded, axis=2))
    A = np.sum(diff_x) + np.sum(diff_y) + np.sum(diff_z)
    
    return A, V


def run_fss_analysis(L_list=[32, 64, 96, 128, 160], realizations=5):
    r"""
    Eksekusi Finite-Size Scaling:
    1. Melacak p_c*(L) adaptif di sekitar persilangan chi = 0 (p in [0.10, 0.16])
    2. Menghitung slope E*(L) = d(ln A) / d(ln V) menggunakan Regresi Lokal
    3. Mengestrapolasi E*_\infty pada 1/L -> 0
    """
    results_FSS = []
    
    print("===========================================================================")
    print("           VALIDASI TAHAP 3: FINITE-SIZE SCALING (L -> \u221e)")
    print("===========================================================================")
    print(f"{'L':<6} | {'1/L':<8} | {'p_c*(L)':<10} | {'E*(L)':<10} | {'STD(E*)':<10} | {'R^2':<8}")
    print("---------------------------------------------------------------------------")

    for L in L_list:
        p_coarse = np.linspace(0.10, 0.16, 25)
        E_samples = []
        pc_samples = []
        r2_samples = []

        for _ in range(realizations):
            # 1. Sweep untuk mencari titik potong chi = 0
            chis, ln_A_list, ln_V_list, p_used = [], [], [], []
            for p in p_coarse:
                grid = (np.random.rand(L, L, L) < p).astype(int)
                chi = compute_euler_exact(grid)
                A, V = measure_area_and_volume(grid)
                
                if A > 0 and V > 0:
                    chis.append(chi)
                    ln_A_list.append(np.log(A))
                    ln_V_list.append(np.log(V))
                    p_used.append(p)

            chis = np.array(chis)
            ln_A_list = np.array(ln_A_list)
            ln_V_list = np.array(ln_V_list)
            p_used = np.array(p_used)

            # 2. Temukan indeks paling dekat dengan chi = 0
            idx_zero = np.argmin(np.abs(chis))
            
            # Estimasi p_c*(L) lokal
            if idx_zero > 0 and idx_zero < len(chis) - 1:
                p1, p2 = p_used[idx_zero-1], p_used[idx_zero+1]
                c1, c2 = chis[idx_zero-1], chis[idx_zero+1]
                p_c_est = p1 - c1 * (p2 - p1) / (c2 - c1) if c2 != c1 else p_used[idx_zero]
            else:
                p_c_est = p_used[idx_zero]

            # 3. Regresi Lokal 7-Titik di sekitar chi = 0 untuk ekstraksi E*
            window = 3 # 3 kiri, 1 tengah, 3 kanan = 7 titik
            start_i = max(0, idx_zero - window)
            end_i = min(len(chis), idx_zero + window + 1)
            
            slope, intercept, r_value, _, _ = linregress(ln_V_list[start_i:end_i], ln_A_list[start_i:end_i])
            
            E_samples.append(slope)
            pc_samples.append(p_c_est)
            r2_samples.append(r_value**2)

        mean_E = np.mean(E_samples)
        std_E = np.std(E_samples)
        mean_pc = np.mean(pc_samples)
        mean_r2 = np.mean(r2_samples)
        inv_L = 1.0 / L

        results_FSS.append({
            'L': L, 'inv_L': inv_L, 'p_c': mean_pc,
            'E_star': mean_E, 'std_E': std_E, 'R2': mean_r2
        })

        print(f"{L:<6} | {inv_L:<8.4f} | {mean_pc:<10.4f} | {mean_E:<10.6f} | {std_E:<10.6f} | {mean_r2:<8.5f}")

    print("===========================================================================")

    # 4. Fit Ekstrapolasi Linier L -> infinity (1/L -> 0)
    inv_L_arr = np.array([r['inv_L'] for r in results_FSS])
    E_star_arr = np.array([r['E_star'] for r in results_FSS])
    
    slope_fss, intercept_fss, r_fss, _, std_err_fss = linregress(inv_L_arr, E_star_arr)
    E_infty = intercept_fss

    print(f"\n[+] HASIL EKSTRAPOLASI LIMIT TERMODINAMIKA (L -> \u221e):")
    print(f"    -> Intersep (E*_\u221e)        = {E_infty:.6f}")
    print(f"    -> Standard Error (E*_\u221e) = {std_err_fss:.6f}")
    print(f"    -> Kemiringan (a)       = {slope_fss:.6f}")
    print(f"    -> R^2 Fitting FSS      = {r_fss**2:.5f}")

    # 5. Visualisasi Plot FSS
    plt.figure(figsize=(9, 6))
    plt.errorbar(inv_L_arr, E_star_arr, yerr=[r['std_E'] for r in results_FSS], 
                 fmt='o', color='blue', ecolor='lightblue', capsize=5, label=r'Data Simulasi $E^*(L)$')
    
    # Garis Fit Ekstrapolasi
    inv_L_fit = np.linspace(0, max(inv_L_arr) * 1.1, 100)
    E_fit = slope_fss * inv_L_fit + intercept_fss
    plt.plot(inv_L_fit, E_fit, 'r--', label=f'Fit Linier: $E^*(L) = {slope_fss:.3f}(1/L) + {E_infty:.4f}$')
    
    # Garis Referensi
    plt.axhline(E_infty, color='green', linestyle=':', label=f'$E^*_\\infty$ Ekstrapolasi = {E_infty:.4f}')
    plt.axhline(0.5, color='orange', linestyle='-.', label=r'Hipotesis $E^* = 1/2$')

    plt.xlim(left=0)
    plt.xlabel(r'Invers Ukuran Kisi ($1/L$)', fontsize=11)
    plt.ylabel(r'Eksponent Topologi $E^*(L)$', fontsize=11)
    plt.title(r'Tahap 3: Finite-Size Scaling $E^*(L)$ vs $1/L$ untuk $L \in \{32, 64, 96, 128, 160\}$', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

    return results_FSS, E_infty

# --- EKSEKUSI LANGSUNG DI SINI ---
if __name__ == "__main__":
    results, E_limit = run_fss_analysis(L_list=[32, 64, 96, 128, 160], realizations=5)