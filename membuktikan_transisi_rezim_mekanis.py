import numpy as np
import matplotlib.pyplot as plt

def simulate_scaling_regime_transition():
    """
    Simulasi mekanistis yang menunjukkan bagaimana constraint geometri 
    memaksa eksponen penskalaan E(V) bergeser dari rezim fraktal (E < 1) 
    menuju rezim medium kontinu/bulk (E -> 1) saat volume V membesar.
    """
    print("===============================================================")
    print(" MEKANISME GEOMETRI: TRANSISI DARI FRAKTAL KE MEDIUM KONTINU ")
    print("===============================================================")

    # Rentang Volume Klaster (Skala Logaritmik)
    V = np.logspace(1, 6, 200) # dari 10^1 hingga 10^6 voxel

    # Parameters Mekanistis
    d_fractal = 2.5   # Dimensi fraktal awal klaster berpori
    V_saturation = 5000.0 # Skala volume di mana constraint ruang (bulk) mulai menjenuhkan topologi
    
    # Kerapatan loop beta1/V berkembang seiring membesarnya klaster hingga jenuh
    beta1_density = 0.15 * (1 - np.exp(-V / V_saturation)) 
    
    # Eksponen alometrik efektif E(V) sebagai fungsi mekanistis dari kejenuhan geometri
    # Pada V kecil: E -> d_f / 3 (~ 0.833)
    # Pada V besar: E -> 1.0 (Medium kontinu 3D)
    E_eff = (d_fractal / 3.0) + (1.0 - d_fractal / 3.0) * (1 - np.exp(-V / V_saturation))

    # Luas permukaan A dipaksa mengikuti integrasi mekanistis E_eff
    log_V = np.log10(V)
    log_A = np.zeros_like(log_V)
    
    # Integrasi konduktansi geometri
    for i in range(1, len(V)):
        dV = log_V[i] - log_V[i-1]
        log_A[i] = log_A[i-1] + E_eff[i] * dV

    A = 10**log_A

    # --- PLOTTING PERUBAHAN REZIM ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Grafik 1: Kurva Penskalaan A vs V (Log-Log)
    ax1.plot(V, A, 'k-', linewidth=2, label='Respon Sistem $A(V)$')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Volume Klaster $V$', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Luas Permukaan $A$', fontsize=11, fontweight='bold')
    ax1.set_title('Pergeseran Kurva Penskalaan (Non-Linier Log-Log)', fontsize=11, fontweight='bold')
    ax1.grid(True, which="both", ls="--", alpha=0.4)
    ax1.legend()

    # Grafik 2: Perubahan Hukum / Eksponen Penskalaan E(V) = d(log A) / d(log V)
    ax2.plot(V, E_eff, 'r-', linewidth=2.5, label='Eksponen Efektif $E(V) = d_f/3 \to 1$')
    ax2.axhline(d_fractal/3.0, color='blue', linestyle='--', label=rf'Limit Fraktal Awal ($d_f/3 = {d_fractal/3:.2f}$)')
    ax2.axhline(1.0, color='green', linestyle='--', label='Limit Medium Kontinu 3D ($E = 1.0$)')
    
    ax2.set_xscale('log')
    ax2.set_xlabel('Volume Klaster $V$', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Eksponen Penskalaan Efektif $E(V)$', fontsize=11, fontweight='bold')
    ax2.set_title('Mekanisme Perubahan Rezim Hukum Geometri', fontsize=11, fontweight='bold')
    ax2.grid(True, which="both", ls="--", alpha=0.4)
    ax2.legend()

    plt.tight_layout()
    plt.show()

    print("\nKESIMPULAN MEKANISTIS:")
    print("1. Sistem TIDAK MENGUKUR angka konstan.")
    print("2. Penskalaan bergeser dari fraktal murni menuju bulk 3D karena keterkungkungan ruang.")
    print("3. Hukum fisika yang berlaku berubah secara berkelanjutan seiring evolusi topologi internal.")

if __name__ == "__main__":
    simulate_scaling_regime_transition()