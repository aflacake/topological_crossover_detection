import numpy as np
import matplotlib.pyplot as plt

def compute_euler_characteristic_exact(grid_3d):
    """
    Menghitung Karakteristik Euler (chi = N0 - N1 + N2 - N3) dari matriks 3D biner.
    Grid biner: 1 = voxel padat, 0 = kosong.
    """
    voxels = np.argwhere(grid_3d == 1)
    if len(voxels) == 0:
        return 0, 0, 0, 0, 0

    N3 = len(voxels)
    vertices = set()
    edges = set()
    faces = set()

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

    N0, N1, N2 = len(vertices), len(edges), len(faces)
    chi = N0 - N1 + N2 - N3
    return chi, N0, N1, N2, N3


def generate_percolation_grid(L, p):
    """Membuat kisi biner 3D berukuran LxLxL dengan probabilitas okupansi p."""
    return (np.random.rand(L, L, L) < p).astype(int)


def run_stage_2_sweep_and_plot(L_list=[16, 24], p_steps=51, realizations=3):
    """
    Memetakan kurva chi(p) dari p = 0.0 sampai 1.0, mencari titik chi = 0,
    dan memunculkan plot grafik.
    
    Catatan: L dikosongkan ke ukuran [16, 24] untuk eksekusi cepat di Python murni,
    namun bisa disesuaikan ke [32, 48] jika menggunakan solver C/C++/Cython.
    """
    p_values = np.linspace(0.0, 1.0, p_steps)
    results = {}

    print(f"{'Scale L':<8} | {'p_c (chi=0) Est.':<18} | {'Min Chi':<12} | {'Max Chi':<12}")
    print("-" * 60)

    plt.figure(figsize=(10, 6))

    for L in L_list:
        chi_means = []
        for p in p_values:
            chi_samples = []
            for _ in range(realizations):
                grid = generate_percolation_grid(L, p)
                chi, _, _, _, _ = compute_euler_characteristic_exact(grid)
                chi_samples.append(chi)
            chi_means.append(np.mean(chi_samples))

        chi_means = np.array(chi_means)
        results[L] = chi_means

        # Cari perkiraan titik p di mana chi melintas 0
        zero_crossings = []
        for i in range(len(chi_means) - 1):
            if chi_means[i] * chi_means[i+1] <= 0:
                # Interpolasi linier lokal untuk menemukan p_c
                p1, p2 = p_values[i], p_values[i+1]
                c1, c2 = chi_means[i], chi_means[i+1]
                if c2 != c1:
                    p_zero = p1 - c1 * (p2 - p1) / (c2 - c1)
                    zero_crossings.append(p_zero)

        p_c_str = ", ".join([f"{pz:.4f}" for pz in zero_crossings]) if zero_crossings else "N/A"
        print(f"L = {L:<4} | {p_c_str:<18} | {np.min(chi_means):<12.1f} | {np.max(chi_means):<12.1f}")

        # Plot kurva untuk kisi L
        plt.plot(p_values, chi_means, marker='o', markersize=3, label=f'L = {L}')

    # Visualisasi Garis Referensi
    plt.axhline(0, color='red', linestyle='--', linewidth=1.5, label=r'$\chi = 0$ (Transisi Topologi)')
    plt.axvline(0.3116, color='gray', linestyle=':', label=r'$p_c$ Literatur 3D ($\approx 0.3116$)')

    plt.title(r'Validasi Tahap 2: Kurva Karakteristik Euler $\chi(p)$ pada Kisi 3D', fontsize=12)
    plt.xlabel('Probabilitas Okupansi ($p$)', fontsize=11)
    plt.ylabel(r'Karakteristik Euler ($\chi$)', fontsize=11)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

    return p_values, results

if __name__ == "__main__":
    # Eksekusi sweep
    p_vals, res = run_stage_2_sweep_and_plot(L_list=[16, 24], p_steps=51, realizations=3)