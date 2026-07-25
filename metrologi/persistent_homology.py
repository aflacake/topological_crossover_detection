import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd

def run_persistent_homology_cubical(L=30, seed=42):
    """
    Menghitung Persistent Homology pada kisi 3D menggunakan Gudhi Cubical Complex.
    Nilai voksel dievaluasi sebagai filtrasi kontinu p in [0, 1].
    """
    np.random.seed(seed)
    
    # 1. Buat matriks filtrasi acak Uniform(0,1) pada kisi L x L x L
    # Setiap voksel memiliki 'waktu muncul' (threshold p) masing-masing
    filtration_grid = np.random.rand(L, L, L)

    print(f"Membangun Cubical Complex 3D berukuran {L}x{L}x{L}...")
    
    # 2. Inisialisasi CubicalComplex dari Gudhi
    cubical_complex = gd.CubicalComplex(
        dimensions=[L, L, L], 
        top_dimensional_cells=filtration_grid.flatten()
    )

    print("Menghitung Persistent Homology (Birth, Death, Persistence)...")
    # Hitung persisten hingga dimensi 2 (beta_0, beta_1, beta_2)
    persistence = cubical_complex.persistence()

    # 3. Pisahkan pasangan (Birth, Death) berdasarkan dimensi H_0, H_1, H_2
    h0 = [] # Komponen
    h1 = [] # Loop / Tunnel
    h2 = [] # Void / Rongga

    for dim, (birth, death) in persistence:
        # Jika death == inf (tidak pernah mati), ganti dengan 1.0 untuk visualisasi
        d_val = 1.0 if np.isinf(death) else death
        
        if dim == 0:
            h0.append((birth, d_val))
        elif dim == 1:
            h1.append((birth, d_val))
        elif dim == 2:
            h2.append((birth, d_val))

    h0 = np.array(h0)
    h1 = np.array(h1)
    h2 = np.array(h2)

    # 4. Visualisasi Persistence Diagrams & Betti Curves dari Barcode
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # --- Subplot (0,0): Persistence Diagram (Birth vs Death) ---
    ax_diag = axes[0, 0]
    if len(h0) > 0: ax_diag.scatter(h0[:, 0], h0[:, 1], c='blue', s=10, alpha=0.5, label=r'$H_0$ (Komponen)')
    if len(h1) > 0: ax_diag.scatter(h1[:, 0], h1[:, 1], c='red', s=10, alpha=0.5, label=r'$H_1$ (Loop/Tunnel)')
    if len(h2) > 0: ax_diag.scatter(h2[:, 0], h2[:, 1], c='green', s=10, alpha=0.5, label=r'$H_2$ (Rongga)')
    
    ax_diag.plot([0, 1], [0, 1], 'k--', alpha=0.5) # Garis diagonal Birth = Death
    ax_diag.axvline(0.3116, color='purple', linestyle='--', label=r'$p_c \approx 0.3116$')
    ax_diag.set_xlabel(r'Birth ($p$)', fontsize=11)
    ax_diag.set_ylabel(r'Death ($p$)', fontsize=11)
    ax_diag.set_title('Persistence Diagram 3D Cubical Complex', fontsize=12)
    ax_diag.legend(loc='lower right')
    ax_diag.grid(True, linestyle='--', alpha=0.5)

    # --- Subplot (0,1): Life Span (Persistence = Death - Birth) ---
    ax_life = axes[0, 1]
    if len(h1) > 0:
        pers_h1 = h1[:, 1] - h1[:, 0]
        ax_life.scatter(h1[:, 0], pers_h1, c='red', s=12, alpha=0.6, label=r'$H_1$ Lifetime')
    if len(h2) > 0:
        pers_h2 = h2[:, 1] - h2[:, 0]
        ax_life.scatter(h2[:, 0], pers_h2, c='green', s=12, alpha=0.6, label=r'$H_2$ Lifetime')
    
    ax_life.axvline(0.3116, color='purple', linestyle='--', label=r'$p_c \approx 0.3116$')
    ax_life.set_xlabel(r'Birth ($p$)', fontsize=11)
    ax_life.set_ylabel('Persistence (Death - Birth)', fontsize=11)
    ax_life.set_title(r'Lifetime Siklus Topologi ($H_1$ dan $H_2$)', fontsize=12)
    ax_life.legend()
    ax_life.grid(True, linestyle='--', alpha=0.5)

    # --- Subplot (1,0): Betti Curves Eksak Kontinu dari Filtrasi ---
    ax_betti = axes[1, 0]
    p_grid = np.linspace(0.01, 0.99, 200)
    
    b0_curve = [np.sum((h0[:, 0] <= p) & (h0[:, 1] > p)) for p in p_grid]
    b1_curve = [np.sum((h1[:, 0] <= p) & (h1[:, 1] > p)) for p in p_grid]
    b2_curve = [np.sum((h2[:, 0] <= p) & (h2[:, 1] > p)) for p in p_grid]

    ax_betti.plot(p_grid, b0_curve, color='blue', label=r'$\beta_0(p)$')
    ax_betti.plot(p_grid, b1_curve, color='red', label=r'$\beta_1(p)$')
    ax_betti.plot(p_grid, b2_curve, color='green', label=r'$\beta_2(p)$')
    ax_betti.axvline(0.3116, color='purple', linestyle='--', label=r'$p_c \approx 0.3116$')
    ax_betti.set_xlabel(r'Filtration Parameter $p$', fontsize=11)
    ax_betti.set_ylabel('Betti Numbers Eksak', fontsize=11)
    ax_betti.set_title('Betti Curves Kontinu dari Persistent Homology', fontsize=12)
    ax_betti.legend()
    ax_betti.grid(True, linestyle='--', alpha=0.5)

    # --- Subplot (1,1): Reconstructed Euler Characteristic Curve ---
    ax_chi = axes[1, 1]
    chi_curve = np.array(b0_curve) - np.array(b1_curve) + np.array(b2_curve)
    ax_chi.plot(p_grid, chi_curve, 'k-', linewidth=2, label=r'$\chi(p) = \beta_0 - \beta_1 + \beta_2$')
    ax_chi.axhline(0, color='gray', linestyle=':')
    ax_chi.axvline(0.3116, color='purple', linestyle='--', label=r'$p_c \approx 0.3116$')
    ax_chi.set_xlabel(r'Filtration Parameter $p$', fontsize=11)
    ax_chi.set_ylabel(r'Karakteristik Euler ($\chi$)', fontsize=11)
    ax_chi.set_title('Karakteristik Euler Kontinu', fontsize=12)
    ax_chi.legend()
    ax_chi.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_persistent_homology_cubical(L=30)