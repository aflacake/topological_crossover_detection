import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from multiprocessing import Pool, cpu_count
import time

def compute_euler_exact(grid_3d):
    """Menghitung Karakteristik Euler eksak (chi = N0 - N1 + N2 - N3)."""
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

def single_sample_worker(args):
    """Fungsi pekerja untuk 1 sampel realisasi acak."""
    L, p = args
    grid = (np.random.rand(L, L, L) < p).astype(int)
    
    struct_6 = np.zeros((3, 3, 3), dtype=int)
    struct_6[1, 1, :] = 1
    struct_6[1, :, 1] = 1
    struct_6[:, 1, 1] = 1

    labeled_grid, num_features = label(grid, structure=struct_6)
    beta_0 = num_features

    if num_features == 0:
        return 0, 0, 0, 0, 0, 0

    # P_max (Fraksi klaster terbesar)
    sizes = np.bincount(labeled_grid.ravel())
    sizes[0] = 0
    p_max = sizes.max() / (L**3)

    # Spanning condition (x=0 ke x=L-1)
    left = set(np.unique(labeled_grid[0, :, :])) - {0}
    right = set(np.unique(labeled_grid[-1, :, :])) - {0}
    is_span = int(len(left.intersection(right)) > 0)

    # Rongga Tertutup (beta_2)
    padded_bg = np.pad(1 - grid, 1, mode='constant', constant_values=1)
    struct_26 = np.ones((3, 3, 3), dtype=int)
    _, num_bg = label(padded_bg, structure=struct_26)
    beta_2 = max(0, num_bg - 1)

    # Euler & beta_1
    chi = compute_euler_exact(grid)
    beta_1 = beta_0 + beta_2 - chi

    return is_span, p_max, beta_0, beta_1, beta_2, chi

def run_high_precision_experiment(L=24, p_steps=40, realizations=200):
    p_arr = np.linspace(0.02, 0.55, p_steps)
    
    pspan_arr = np.zeros(p_steps)
    pmax_arr = np.zeros(p_steps)
    b0_arr = np.zeros(p_steps)
    b1_arr = np.zeros(p_steps)
    b2_arr = np.zeros(p_steps)
    chi_arr = np.zeros(p_steps)

    num_cores = cpu_count()
    print(f"Running High-Precision Ensemble: L={L}, N_samples={realizations} per titik")
    print(f"Menggunakan {num_cores} CPU cores multiprocessing...")
    
    start_time = time.time()

    for idx, p in enumerate(p_arr):
        tasks = [(L, p) for _ in range(realizations)]
        
        with Pool(processes=num_cores) as pool:
            results = pool.map(single_sample_worker, tasks)
        
        results = np.array(results)
        
        pspan_arr[idx] = np.mean(results[:, 0])
        pmax_arr[idx]  = np.mean(results[:, 1])
        b0_arr[idx]    = np.mean(results[:, 2])
        b1_arr[idx]    = np.mean(results[:, 3])
        b2_arr[idx]    = np.mean(results[:, 4])
        chi_arr[idx]   = np.mean(results[:, 5])

        if (idx + 1) % 10 == 0 or idx == p_steps - 1:
            print(f"Progress: {idx+1}/{p_steps} p-points diselesaikan...")

    print(f"Selesai dalam {time.time() - start_time:.2f} detik.")

    # Plot Hasil Kualitas Tinggi
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Plot 1: Sigmoid Mulus P_span & P_max
    ax1.plot(p_arr, pspan_arr, 'o-', color='crimson', linewidth=2, label=r'$P_{\text{span}}$ (Sigmoid Mulus)')
    ax1.plot(p_arr, pmax_arr, 's-', color='darkblue', linewidth=2, label=r'$P_{\text{max}}$ (Fraksi Klaster Utama)')
    ax1.axvline(0.3116, color='purple', linestyle='--', label=r'$p_c \approx 0.3116$')
    ax1.set_ylabel('Perkolasi Makroskopis', fontsize=11)
    ax1.set_title(f'Statistik Ansambel Mulus ($L={L}, N={realizations}$ realisasi)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper left')

    # Plot 2: Betti Numbers
    ax2.plot(p_arr, b0_arr, 'o-', color='blue', label=r'$\beta_0$ (Komponen)')
    ax2.plot(p_arr, b1_arr, 's-', color='red', label=r'$\beta_1$ (Loop/Tunnel)')
    ax2.plot(p_arr, b2_arr, '^-', color='green', label=r'$\beta_2$ (Rongga)')
    ax2.axvline(0.13, color='orange', linestyle=':', linewidth=2, label=r'Cross-over $\beta_1 = \beta_0$')
    ax2.axvline(0.3116, color='purple', linestyle='--')
    ax2.set_ylabel('Jumlah Topologi', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right')

    # Plot 3: Karakteristik Euler
    ax3.plot(p_arr, chi_arr, 'k-', linewidth=2, label=r'$\chi(p) = \beta_0 - \beta_1 + \beta_2$')
    ax3.axhline(0, color='gray', linestyle=':')
    ax3.axvline(0.13, color='orange', linestyle=':', linewidth=2, label=r'Zero Crossing Pertama ($\chi=0$)')
    ax3.axvline(0.3116, color='purple', linestyle='--', label=r'$p_c$ (Lembah $\chi$ Minimum)')
    ax3.set_xlabel('Probabilitas Okupansi ($p$)', fontsize=11)
    ax3.set_ylabel('Karakteristik Euler ($\chi$)', fontsize=11)
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend(loc='lower right')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Menjalankan 200 realisasi per titik p
    run_high_precision_experiment(L=24, p_steps=40, realizations=200)