import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, distance_transform_edt
import gudhi as gd
import time

def compute_correlation_length(grid_3d, labeled_grid, num_features):
    """
    Menghitung Panjang Korelasi xi(p) dari radius girus (radius of gyration) klaster.
    xi^2 = 2 * sum(S_i^2 * R_g,i^2) / sum(S_i^2)  (mengabaikan klaster terbesar/spanning)
    """
    if num_features <= 1:
        return 0.0

    sizes = np.bincount(labeled_grid.ravel())
    sizes[0] = 0  # Ignore background
    
    # Abaikan klaster terbesar untuk menghitung korelasi fluktuasi
    max_label = sizes.argmax()
    
    sum_s2_rg2 = 0.0
    sum_s2 = 0.0

    for l in range(1, num_features + 1):
        if l == max_label:
            continue
        
        coords = np.argwhere(labeled_grid == l)
        s_i = len(coords)
        if s_i < 2:
            continue
            
        com = coords.mean(axis=0)
        rg2 = np.mean(np.sum((coords - com)**2, axis=1))
        
        sum_s2_rg2 += (s_i**2) * rg2
        sum_s2 += (s_i**2)

    if sum_s2 == 0:
        return 0.0
    
    return np.sqrt(sum_s2_rg2 / sum_s2)

def run_grand_unified_experiment(L=28, p_steps=50, realizations=10):
    p_arr = np.linspace(0.01, 0.99, p_steps)
    
    # Array Penyimpan Data
    b0_arr = np.zeros(p_steps)
    b1_arr = np.zeros(p_steps)
    b2_arr = np.zeros(p_steps)
    chi_arr = np.zeros(p_steps)
    smax_arr = np.zeros(p_steps)
    pspan_arr = np.zeros(p_steps)
    xi_arr = np.zeros(p_steps)

    print(f"==================================================")
    print(f"Menjalankan Grand Unified Experiment (L={L}, Steps={p_steps})")
    print(f"==================================================")
    
    start_time = time.time()

    struct_6 = np.zeros((3, 3, 3), dtype=int)
    struct_6[1, 1, :] = 1
    struct_6[1, :, 1] = 1
    struct_6[:, 1, 1] = 1

    for idx, p in enumerate(p_arr):
        b0_s, b1_s, b2_s, chi_s = [], [], [], []
        smax_s, pspan_s, xi_s = [], [], []

        for _ in range(realizations):
            # 1. Generate Grid
            grid = (np.random.rand(L, L, L) < p).astype(int)
            
            # 2. Topologi via Gudhi CubicalComplex
            cc = gd.CubicalComplex(dimensions=[L, L, L], top_dimensional_cells=grid.flatten())
            # Trik filtrasi biner: 1 saat muncul, inf saat kosong
            filt = np.where(grid.flatten() == 1, 0.0, 2.0)
            cc_bin = gd.CubicalComplex(dimensions=[L, L, L], top_dimensional_cells=filt)
            p_res = cc_bin.persistence()
            
            # Hitung Betti numbers pada threshold p
            b0 = sum(1 for dim, (b, d) in p_res if dim == 0 and b <= 0.5 and d > 0.5)
            b1 = sum(1 for dim, (b, d) in p_res if dim == 1 and b <= 0.5 and d > 0.5)
            b2 = sum(1 for dim, (b, d) in p_res if dim == 2 and b <= 0.5 and d > 0.5)
            chi = b0 - b1 + b2

            # 3. Statistik Perkolasi
            labeled, num_feat = label(grid, structure=struct_6)
            if num_feat > 0:
                sizes = np.bincount(labeled.ravel())
                sizes[0] = 0
                smax = sizes.max() / (L**3) # Normalisasi volume
                
                left = set(np.unique(labeled[0, :, :])) - {0}
                right = set(np.unique(labeled[-1, :, :])) - {0}
                is_span = int(len(left.intersection(right)) > 0)
                
                xi = compute_correlation_length(grid, labeled, num_feat)
            else:
                smax, is_span, xi = 0, 0, 0

            b0_s.append(b0); b1_s.append(b1); b2_s.append(b2); chi_s.append(chi)
            smax_s.append(smax); pspan_s.append(is_span); xi_s.append(xi)

        b0_arr[idx] = np.mean(b0_s)
        b1_arr[idx] = np.mean(b1_s)
        b2_arr[idx] = np.mean(b2_s)
        chi_arr[idx] = np.mean(chi_s)
        smax_arr[idx] = np.mean(smax_s)
        pspan_arr[idx] = np.mean(pspan_s)
        xi_arr[idx] = np.mean(xi_s)

    print(f"Eksperimen Selesai dalam {time.time() - start_time:.2f} detik.")

    # --- CARI TITIK-TITIK TRANSISI UNTUK GARIS VERTIKAL ---
    # 1. Zero crossing chi pertama dan kedua
    zero_crossings = np.where(np.diff(np.sign(chi_arr)))[0]
    p_chi_zero_1 = p_arr[zero_crossings[0]] if len(zero_crossings) > 0 else 0.13
    p_chi_zero_2 = p_arr[zero_crossings[1]] if len(zero_crossings) > 1 else 0.62

    # 2. Point where P_span = 50% (0.5)
    idx_p50 = np.argmin(np.abs(pspan_arr - 0.5))
    p_span_50 = p_arr[idx_p50]

    # --- PLOTTING MASTER CHART ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

    # Subplot 1: Perkolasi Makroskopis (S_max, P_span, Correlation Length xi)
    ax1 = axes[0]
    ax1.plot(p_arr, pspan_arr, 'r-o', linewidth=2, label=r'$P_{\text{span}}$ (Probabilitas Spanning)')
    ax1.plot(p_arr, smax_arr, 'b-s', linewidth=2, label=r'$S_{\text{max}} / L^3$ (Fraksi Klaster Terbesar)')
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(
        p_arr, 
        xi_arr, 
        color='m', 
        marker='^', 
        linestyle='--', 
        linewidth=1.5, 
        label=r'$\xi(p)$ (Panjang Korelasi)'
    )
    ax1_twin.set_ylabel(r'Panjang Korelasi $\xi$', color='m', fontsize=11)
    
    ax1.set_ylabel('Besaran Perkolasi Global', fontsize=11)
    ax1.set_title(r'Peta Fase Topologi & Fisik Perkolasi 3D ($L={L}$, $0 \leq p \leq 1$)', fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')

    # Subplot 2: Betti Numbers Decompositions
    ax2 = axes[1]
    ax2.plot(p_arr, b0_arr, 'blue', linewidth=2, label=r'$\beta_0$ (Komponen Terpisah)')
    ax2.plot(p_arr, b1_arr, 'red', linewidth=2, label=r'$\beta_1$ (Loop / Terowongan)')
    ax2.plot(p_arr, b2_arr, 'green', linewidth=2, label=r'$\beta_2$ (Rongga / Void)')
    ax2.set_ylabel('Jumlah Betti Numbers', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend(loc='upper right')

    # Subplot 3: Karakteristik Euler Chi(p)
    ax3 = axes[2]
    ax3.plot(p_arr, chi_arr, 'k-', linewidth=2.5, label=r'$\chi(p) = \beta_0 - \beta_1 + \beta_2$')
    ax3.axhline(0, color='black', linestyle=':', linewidth=1)
    ax3.set_xlabel(r'Probabilitas Okupansi ($p$)', fontsize=12, fontweight='bold')
    ax3.set_ylabel(r'Karakteristik Euler ($\chi$)', fontsize=11)
    ax3.grid(True, linestyle='--', alpha=0.4)
    ax3.legend(loc='lower right')

    # --- TAMBAHKAN GARIS VERTIKAL PENANDA DI SEMUA SUBPLOT ---
    vlines = [
        (p_chi_zero_1, 'orange', ':', r'1. $\chi = 0$ Pertama (p ≈ {p_chi_zero_1:.2f})'),
        (p_span_50, 'cyan', '-.', r'2. $P_{{span}} = 50\%$ (p ≈ {p_span_50:.2f})'),
        (0.3116, 'purple', '--', r'3. $p_c$ Literatur (≈ 0.3116)'),
        (p_chi_zero_2, 'brown', ':', r'4. $\chi = 0$ Kedua (p ≈ {p_chi_zero_2:.2f})')
    ]

    for ax in [ax1, ax2, ax3]:
        for x_val, color, style, label_text in vlines:
            ax.axvline(x_val, color=color, linestyle=style, linewidth=1.8)

    # Tambahkan Legend Khusus Garis Vertikal di Gambar Atas
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=c, lw=2, ls=s) for _, c, s, _ in vlines]
    custom_labels = [lbl for _, _, _, lbl in vlines]
    ax1.legend(custom_lines, custom_labels, loc='center left', bbox_to_anchor=(0.01, 0.5), title="Garis Transisi Kritis")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_grand_unified_experiment(L=28, p_steps=45, realizations=8)