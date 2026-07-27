import numpy as np

def find_inflection_points(V, beta1):
    """
    Implementasi Metode Konvensional: d^2(beta_1) / dV^2 = 0
    Mencari titik infleksi berdasarkan turunan kedua.
    """
    # Menghitung turunan pertama
    d_beta1 = np.gradient(beta1, V)
    # Menghitung turunan kedua (kelengkungan kurva)
    d2_beta1 = np.gradient(d_beta1, V)
    
    # Mencari titik di mana turunan kedua memotong angka nol (zero-crossings)
    zero_crossings = np.where(np.diff(np.sign(d2_beta1)))[0]
    
    return V[zero_crossings]

if __name__ == "__main__":
    # Simulasi trajektori pertumbuhan spasial realistis (non-sigmoidal/fluktuatif)
    np.random.seed(42)
    V = np.linspace(10, 100, 100)
    # Membentuk kurva Betti Number dengan fluktuasi (noise) lokal
    beta1 = 1 / (1 + np.exp(-(V - 50) / 5)) + np.random.normal(0, 0.015, size=V.shape)

    inflection_points = find_inflection_points(V, beta1)
    print("=== METODE KONVENSIONAL ===")
    print(f"Kriteria: d^2(beta_1)/dV^2 = 0")
    print(f"Jumlah kandidat crossover yang ditemukan: {len(inflection_points)} titik.")
    print("Hasil: Gagal/Ambigu. Kriteria konvensional menghasilkan banyak titik (N >= 3) akibat fluktuasi lokal.")
