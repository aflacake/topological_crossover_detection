import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, binary_dilation, generate_binary_structure
from scipy.spatial.distance import cdist

def measure_true_self_contact_growth(L=32, p_initial=0.25, delta_p=0.03, realizations=4):
    """
    Mengekstrak P(self-contact after growth | r) secara sah dengan menyimulasikan 
    pertumbuhan dinamis klaster (p_initial -> p_initial + delta_p) dan mencatat 
    peristiwa penyatuan dua cabang terpisah yang berjarak r di udara.
    """
    print("===============================================================")
    print("  PENGUKURAN VALID: P(self-contact after growth | r) DINAMIS  ")
    print("===============================================================")

    struct_6 = generate_binary_structure(3, 1)
    
    r_bins = np.linspace(2.0, 10.0, 9)
    total_pairs = np.zeros(len(r_bins) - 1)
    contact_events = np.zeros(len(r_bins) - 1)

    for real in range(realizations):
        # 1. State St: Sistem awal pada p_initial
        grid_t = (np.random.rand(L, L, L) < p_initial).astype(int)
        labeled_t, num_c = label(grid_t, structure=struct_6)

        # 2. State St+1: Pertumbuhan sistem akibat penambahan probabilitas delta_p
        growth_mask = (np.random.rand(L, L, L) < delta_p).astype(int)
        grid_t1 = np.clip(grid_t + growth_mask, 0, 1)
        labeled_t1, _ = label(grid_t1, structure=struct_6)

        for cid in range(1, num_c + 1):
            mask_t = (labeled_t == cid)
            if np.sum(mask_t) < 40: # Filter klaster terlalu kecil
                continue

            # Cari permukaan klaster pada St
            dil_t = binary_dilation(mask_t, structure=struct_6)
            surface_coords = np.argwhere(dil_t & (~mask_t))

            if len(surface_coords) < 10:
                continue

            # Hitung jarak Euclidean r di udara antar semua titik permukaan
            dist_matrix = cdist(surface_coords, surface_coords)

            # Iterasi pasangan permukaan yang dipisahkan oleh void space (r >= 2.0)
            for i in range(len(r_bins) - 1):
                r_min, r_max = r_bins[i], r_bins[i+1]
                pair_mask = (dist_matrix >= r_min) & (dist_matrix < r_max)

                # Pasangan valid pada jarak r
                valid_pairs = np.argwhere(np.triu(pair_mask, k=1))
                total_pairs[i] += len(valid_pairs)

                # Cek apakah setelah pertumbuhan (St+1), kedua titik ini terhubung
                # oleh voxel pertumbuhan baru (Self-Contact Event)
                for p_idx1, p_idx2 in valid_pairs:
                    pt1 = surface_coords[p_idx1]
                    pt2 = surface_coords[p_idx2]

                    # Periksa apakah pt1 dan pt2 sekarang berada dalam klaster t1 yang sama
                    # dan terhubung melalui jalur voxel baru di grid_t1
                    cid_t1_pt1 = labeled_t1[pt1[0], pt1[1], pt1[2]]
                    cid_t1_pt2 = labeled_t1[pt2[0], pt2[1], pt2[2]]

                    if cid_t1_pt1 > 0 and cid_t1_pt1 == cid_t1_pt2:
                        contact_events[i] += 1

    # Probabilitas Dinamis Sejati
    p_growth_contact = np.divide(
        contact_events, 
        total_pairs, 
        out=np.zeros_like(contact_events, dtype=float), 
        where=total_pairs > 0
    )

    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(r_centers, p_growth_contact, 's-', color='darkgreen', linewidth=2, label=r'$P(\mathrm{self\text{-}contact\ after\ growth} \mid r)$')
    plt.xlabel(r'Jarak Spasial Antar-Cabang di Udara $r$ (voxel)', fontsize=11, fontweight='bold')
    plt.ylabel(r'Probabilitas Penyatuan Dinamis $P(r)$', fontsize=11, fontweight='bold')
    plt.title(r'Probabilitas Self-Contact Dinamis Pasca-Pertumbuhan ($\Delta p$)', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

    print("\nHASIL PENGUKURAN DINAMIS YANG VALID:")
    for r_v, p_v, n_p in zip(r_centers, p_growth_contact, total_pairs):
        print(f"• Jarak r = {r_v:.2f} voxel | P_growth_contact = {p_v:.5f} (diuji dari {int(n_p)} pasangan)")

if __name__ == "__main__":
    measure_true_self_contact_growth(L=28, p_initial=0.26, delta_p=0.04, realizations=3)