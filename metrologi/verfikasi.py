import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

def compute_cubical_euler_characteristic(binary_grid):
    """
    Menhitung Karakteristik Euler (chi) sejati pada kisi kubus 3D 
    menggunakan teorema kompleks kubikal (Cubical Complex):
    chi = N_voxels - N_faces + N_edges - N_nodes
    """
    # 1. Jumlah Voxel (3D Cells)
    n_voxels = np.sum(binary_grid)
    if n_voxels == 0:
        return 0, 0, 0
    
    # Pad dengan kondisi batas periodik / nol untuk analisis konektivitas
    padded = np.pad(binary_grid, pad_width=1, mode='wrap')
    
    # 2. Jumlah Muka / Faces (2D Interfaces internal antar Voxel terisi)
    # Muka internal dihitung dari pasangan voxel berdampingan
    fx = np.sum(padded[:-1, :, :] & padded[1:, :, :])
    fy = np.sum(padded[:, :-1, :] & padded[:, 1:, :])
    fz = np.sum(padded[:, :, :-1] & padded[:, :, 1:])
    n_faces = fx + fy + fz
    
    # 3. Jumlah Rusuk / Edges (1D Interfaces)
    # Kuadrat 2x2 voxel yang terisi
    e_xy = np.sum(padded[:-1, :-1, :] & padded[1:, :-1, :] & padded[:-1, 1:, :] & padded[1:, 1:, :])
    e_xz = np.sum(padded[:-1, :, :-1] & padded[1:, :, :-1] & padded[:-1, :, 1:] & padded[1:, :, 1:])
    e_yz = np.sum(padded[:, :-1, :-1] & padded[:, 1:, :-1] & padded[:, :-1, 1:] & padded[:, 1:, 1:])
    n_edges = e_xy + e_xz + e_yz
    
    # 4. Jumlah Titik Sudut / Nodes (0D Interfaces)
    # Kubus 2x2x2 voxel
    n_nodes = np.sum(
        padded[:-1, :-1, :-1] & padded[1:, :-1, :-1] & padded[:-1, 1:, :-1] & padded[1:, 1:, :-1] &
        padded[:-1, :-1, 1:]  & padded[1:, :-1, 1:]  & padded[:-1, 1:, 1:]  & padded[1:, 1:, 1:]
    )
    
    # Formula Euler untuk Kompleks Kubikal
    chi = n_voxels - n_faces + n_edges - n_nodes
    
    # Luas Permukaan Tersebar (Boundary Area)
    # Voxel faces yang terekspos ke ruang kosong
    diff_x = np.abs(np.diff(np.pad(binary_grid, ((1,1),(0,0),(0,0)), mode='wrap'), axis=0))
    diff_y = np.abs(np.diff(np.pad(binary_grid, ((0,0),(1,1),(0,0)), mode='wrap'), axis=1))
    diff_z = np.abs(np.diff(np.pad(binary_grid, ((0,0),(0,0),(1,1)), mode='wrap'), axis=2))
    area = np.sum(diff_x) + np.sum(diff_y) + np.sum(diff_z)
    
    return n_voxels, area, chi

def run_precision_verification(grid_size=64, num_steps=80):
    r"""
    Simulasi verifikasi K_trans = d(ln A) / d(ln V) pada titik kelintasan nol topologis (\chi = 0).
    """
    print("=== MEMULAI SIMULASI PRESISI TOPOLOGIS 3D ===")
    
    grid_prob = np.random.rand(grid_size, grid_size, grid_size)
    p_values = np.linspace(0.10, 0.50, num_steps)
    
    volumes = []
    areas = []
    eulers = []
    p_recorded = []
    
    for p in p_values:
        binary_grid = grid_prob <= p
        V, A, chi = compute_cubical_euler_characteristic(binary_grid)
        
        if V > 0 and A > 0:
            volumes.append(V)
            areas.append(A)
            eulers.append(chi)
            p_recorded.append(p)
            
    volumes = np.array(volumes)
    areas = np.array(areas)
    eulers = np.array(eulers)
    p_recorded = np.array(p_recorded)
    
    # Hitung Logaritma Alometrik
    ln_V = np.log(volumes)
    ln_A = np.log(areas)
    
    # Turunan Logaritmik Presisi Tinggi (Central Difference Gradient)
    K_trans_curve = np.gradient(ln_A, ln_V)
    
    # Deteksi Titik Persilangan Nol Euler (\chi = 0)
    zero_crossing_idx = np.where(np.diff(np.signbit(eulers)))[0]
    
    print("\n=== HASIL EKSPERIMEN NUMERIK PRESISI ===")
    if len(zero_crossing_idx) > 0:
        idx = zero_crossing_idx[0]
        K_val_at_zero = K_trans_curve[idx]
        p_crit = p_recorded[idx]
        
        print(f"Ambang Perkolasi Kritis (p_crit)  : {p_crit:.4f}")
        print(f"Karakteristik Euler (\\chi)        : {eulers[idx]} -> mendekati 0")
        print(f"Nilai Terukur K_trans (dLnA/dLnV) : {K_val_at_zero:.4f}")
        print(f"Nilai Prediksi Teoretis           : 0.8400")
        print(f"Deviasi / Error                   : {abs(K_val_at_zero - 0.8400) / 0.8400 * 100:.2f}%")
    else:
        print(r"Sistem belum mencapai \chi = 0. Tingkatkan grid_size.")

    # Visualisasi
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Volume Terisi ln(V)')
    ax1.set_ylabel(r'Elastisitas Alometrik $K_{trans} = \frac{d(\ln A)}{d(\ln V)}$', color=color)
    ax1.plot(ln_V, K_trans_curve, color=color, linewidth=2, label=r'$K_{trans}$ Terukur')
    ax1.axhline(y=0.84, color='r', linestyle='--', label=r'Prediksi Teoretis ($0.84$)')
    ax1.axhline(y=0.667, color='g', linestyle=':', label=r'Batas Euclidean 3D ($0.67$)')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:orange'
    ax2.set_ylabel(r'Karakteristik Euler ($\chi$)', color=color)
    ax2.plot(ln_V, eulers, color=color, linestyle='-.', label=r'Euler ($\chi$)')
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.tick_params(axis='y', labelcolor=color)

    if len(zero_crossing_idx) > 0:
        ax1.axvline(x=ln_V[zero_crossing_idx[0]], color='purple', alpha=0.5, linestyle='-', label=r'Titik $\chi = 0$')

    fig.tight_layout()
    plt.title(r'Pengujian Verifikasi $K_{trans}$ pada Titik Kelintasan Nol Topologis ($\chi = 0$)')
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run_precision_verification(grid_size=64, num_steps=80)