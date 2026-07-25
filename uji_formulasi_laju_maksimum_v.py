import numpy as np
import matplotlib.pyplot as plt

def test_crossover_definition():
    print("===============================================================")
    print("  UJI FORMULASI MATEMATIS: V* SEBAGAI MAX(dβ1/dV)  ")
    print("===============================================================")

    # Rentang Volume Klaster V
    V = np.linspace(10, 1000, 1000)
    dV = V[1] - V[0]

    # --- KASUS 1: Kurva Sigmoid (Model Saturasi Sederhana) ---
    # β1(V) = β_max / (1 + exp(-(V - V_mid)/scale))
    beta1_sigmoid = 150 / (1 + np.exp(-(V - 400) / 80))

    # --- KASUS 2: Kurva Non-Sigmoid / Bimodal (Ada Fluktuasi / Multi Inflection) ---
    # Memiliki beberapa titik belok d²β1/dV² = 0, tetapi hanya 1 puncak laju utama
    beta1_complex = (
        120 / (1 + np.exp(-(V - 500) / 90)) + 
        15 * np.sin(V / 50) * np.exp(-V / 300)
    )

    # Hitung Turunan Pertama (Laju dβ1/dV) dan Turunan Kedua (d²β1/dV²)
    d1_sig = np.gradient(beta1_sigmoid, dV)
    d2_sig = np.gradient(d1_sig, dV)

    d1_comp = np.gradient(beta1_complex, dV)
    d2_comp = np.gradient(d1_comp, dV)

    # Cari Titik Crossover V* berdasarkan DEFINISI LAJU MAKSIMUM: argmax(dβ1/dV)
    idx_star_sig = np.argmax(d1_sig)
    V_star_sig = V[idx_star_sig]

    idx_star_comp = np.argmax(d1_comp)
    V_star_comp = V[idx_star_comp]

    # Cari semua titik belok d²β1/dV² = 0 (penyilangan nol / zero-crossing)
    zero_crossings_comp = np.where(np.diff(np.sign(d2_comp)))[0]

    # --- VISUALISASI ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # Subplot 1: Kasus Sigmoid β1(V)
    axes[0, 0].plot(V, beta1_sigmoid, 'b-', linewidth=2, label=r'$\beta_1(V)$ Sigmoid')
    axes[0, 0].axvline(V_star_sig, color='red', linestyle='--', label=rf'Titik Crossover $V^* = {V_star_sig:.1f}$')
    axes[0, 0].set_ylabel(r'Jumlah Loop $\beta_1$', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Kasus 1: Kurva Sigmoid Ideal', fontsize=11, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.4)
    axes[0, 0].legend()

    # Subplot 2: Turunan Kasus Sigmoid
    axes[1, 0].plot(V, d1_sig, 'g-', linewidth=2, label=r'Laju $\frac{d\beta_1}{dV}$')
    axes[1, 0].plot(V, d2_sig * 100, 'm--', label=r'Turunan Kedua $\frac{d^2\beta_1}{dV^2} \times 100$')
    axes[1, 0].axvline(V_star_sig, color='red', linestyle='--')
    axes[1, 0].axhline(0, color='black', linewidth=0.8)
    axes[1, 0].set_xlabel('Volume Klaster $V$', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Laju & Akselerasi', fontsize=11, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.4)
    axes[1, 0].legend()

    # Subplot 3: Kasus Non-Sigmoid Kompleks β1(V)
    axes[0, 1].plot(V, beta1_complex, 'b-', linewidth=2, label=r'$\beta_1(V)$ Non-Sigmoid')
    axes[0, 1].axvline(V_star_comp, color='red', linestyle='--', label=rf'Puncak Laju Sejati $V^* = {V_star_comp:.1f}$')
    axes[0, 1].set_ylabel(r'Jumlah Loop $\beta_1$', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Kasus 2: Kurva Non-Sigmoid Kompleks', fontsize=11, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.4)
    axes[0, 1].legend()

    # Subplot 4: Turunan Kasus Non-Sigmoid
    axes[1, 1].plot(V, d1_comp, 'g-', linewidth=2, label=r'Laju $\frac{d\beta_1}{dV}$')
    axes[1, 1].plot(V, d2_comp * 100, 'm--', label=r'Turunan Kedua $\frac{d^2\beta_1}{dV^2} \times 100$')
    axes[1, 1].axvline(V_star_comp, color='red', linestyle='--', label=rf'Puncak Sejati ($V^* = {V_star_comp:.1f}$)')
    
    # Plot semua titik di mana d²β1/dV² = 0
    for idx in zero_crossings_comp:
        axes[1, 1].plot(V[idx], d1_comp[idx], 'ro', markersize=6)
    
    axes[1, 1].axhline(0, color='black', linewidth=0.8)
    axes[1, 1].set_xlabel('Volume Klaster $V$', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Laju & Akselerasi', fontsize=11, fontweight='bold')
    axes[1, 1].set_title(r'Banyak titik $\frac{d^2\beta_1}{dV^2} = 0$ (lingkaran merah), tetapi $V^*$ unik!', fontsize=10, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.4)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()

    print("\nHASIL ANALISIS PENGUJIANKU:")
    print(f"• Kasus 1 (Sigmoid)     : V* unik ditemukan pada V = {V_star_sig:.1f}")
    print(f"• Kasus 2 (Non-Sigmoid) : Ditemukan {len(zero_crossings_comp)} titik di mana d²β1/dV² = 0.")
    print(f"                         Namun titik Crossover Fisis Sejati MAX(dβ1/dV) HANYA ADA 1 yaitu pada V = {V_star_comp:.1f}.")

if __name__ == "__main__":
    test_crossover_definition()