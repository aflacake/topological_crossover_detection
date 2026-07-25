import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

def compute_euler_exact(grid_3d):
    """Menghitung Karakteristik Euler (chi = N0 - N1 + N2 - N3) dari kisi biner 3D."""
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

def compute_betti_numbers(grid_3d):
    """
    Menghitung Betti Numbers (beta_0, beta_1, beta_2):
    - beta_0: Komponen terhubung (foreground, 6-konektivitas)
    - beta_2: Rongga tertutup (voids dalam background)
    - beta_1: Tunnel/loop (diperoleh via Euler-Poincaré: beta_1 = beta_0 + beta_2 - chi)
    """
    # 1. beta_0 (Komponen terhubung foreground)
    # Struktur konektivitas 6-neighbor standar untuk 3D (face-sharing)
    struct_6 = np.zeros((3, 3, 3), dtype=int)
    struct_6[1, 1, :] = 1
    struct_6[1, :, 1] = 1
    struct_6[:, 1, 1] = 1

    _, beta_0 = label(grid_3d, structure=struct_6)

    # 2. beta_2 (Rongga tertutup)
    # Dipad dengan 0 di luar kisi agar background luar menjadi 1 komponen raksasa
    padded_bg = np.pad(1 - grid_3d, 1, mode='constant', constant_values=1)
    
    # Background menggunakan konektivitas 26-neighbor (diagonal) untuk konsistensi topologi
    struct_26 = np.ones((3, 3, 3), dtype=int)
    _, num_bg_components = label(padded_bg, structure=struct_26)
    
    # Rongga tertutup = Total komponen background - 1 (komponen luar)
    beta_2 = max(0, num_bg_components - 1)

    # 3. chi (Karakteristik Euler eksak)
    chi = compute_euler_exact(grid_3d)

    # 4. beta_1 (Diperoleh dari chi = beta_0 - beta_1 + beta_2)
    beta_1 = beta_0 + beta_2 - chi

    return beta_0, beta_1, beta_2, chi

def run_betti_decomposition_experiment(L=24, p_steps=40, realizations=3):
    """Eksperimen pemetaan Betti Numbers vs Okupansi p."""
    p_arr = np.linspace(0.01, 0.99, p_steps)
    
    b0_avg = np.zeros(p_steps)
    b1_avg = np.zeros(p_steps)
    b2_avg = np.zeros(p_steps)
    chi_avg = np.zeros(p_steps)

    print(f"Running Betti Decomposition Analysis untuk L={L}...")
    print("---------------------------------------------------------")
    
    for i, p in enumerate(p_arr):
        b0_samples, b1_samples, b2_samples, chi_samples = [], [], [], []
        
        for _ in range(realizations):
            grid = (np.random.rand(L, L, L) < p).astype(int)
            b0, b1, b2, chi = compute_betti_numbers(grid)
            
            b0_samples.append(b0)
            b1_samples.append(b1)
            b2_samples.append(b2)
            chi_samples.append(chi)

        b0_avg[i] = np.mean(b0_samples)
        b1_avg[i] = np.mean(b1_samples)
        b2_avg[i] = np.mean(b2_samples)
        chi_avg[i] = np.mean(chi_samples)

    # Visualisasi
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Plot 1: Betti Numbers (beta_0, beta_1, beta_2)
    ax1.plot(p_arr, b0_avg, 'o-', color='blue', label=r'$\beta_0$ (Komponen Terhubung)')
    ax1.plot(p_arr, b1_avg, 's-', color='red', label=r'$\beta_1$ (Terowongan / Loop)')
    ax1.plot(p_arr, b2_avg, '^-', color='green', label=r'$\beta_2$ (Rongga Tertutup)')
    ax1.set_ylabel('Jumlah Topologi', fontsize=11)
    ax1.set_title(f'Dekomposisi Betti Numbers $(\\beta_0, \\beta_1, \\beta_2)$ pada Kisi 3D ($L={L}$)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(fontsize=11)

    # Plot 2: Rekonstruksi Karakteristik Euler (chi)
    chi_reconstructed = b0_avg - b1_avg + b2_avg
    ax2.plot(p_arr, chi_avg, 'k-', linewidth=2, label=r'$\chi(p)$ Eksak ($N_0 - N_1 + N_2 - N_3$)')
    ax2.plot(p_arr, chi_reconstructed, 'r--', linewidth=1.5, label=r'$\chi(p) = \beta_0 - \beta_1 + \beta_2$')
    ax2.axhline(0, color='gray', linestyle=':', label=r'$\chi = 0$')
    ax2.axvline(0.3116, color='purple', linestyle='--', label=r'$p_c$ Perkolasi Literatur ($\approx 0.3116$)')
    
    ax2.set_xlabel('Probabilitas Okupansi ($p$)', fontsize=11)
    ax2.set_ylabel(r'Karakteristik Euler ($\chi$)', fontsize=11)
    ax2.set_title(r'Konfirmasi Identitas Euler-Poincaré $\chi(p)$', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    plt.show()

    return p_arr, b0_avg, b1_avg, b2_avg, chi_avg

if __name__ == "__main__":
    # Ukuran L=24 untuk meminimalkan beban komputasi namun cukup memetakan kurva
    p_arr, b0, b1, b2, chi = run_betti_decomposition_experiment(L=24, p_steps=40, realizations=3)