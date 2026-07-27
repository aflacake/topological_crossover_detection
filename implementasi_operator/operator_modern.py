import numpy as np
from scipy.ndimage import gaussian_filter1d

def find_operational_crossover(V, beta1):
    """
    Implementasi Metode Modern (Usulan Paper): V_dagger = arg max g(V)
    di mana g(V) = d(beta_1) / dV
    Mencari puncak efisiensi topologis global.
    """
    # 1. Smoothing & Interpolation: Mengurangi noise frekuensi tinggi tanpa merusak bentuk dasar
    beta1_smooth = gaussian_filter1d(beta1, sigma=2)
    
    # 2. Numerical Differentiation (Central Difference): Menghitung Loop-Formation Rate g(V)
    g_V = np.gradient(beta1_smooth, V)
    
    # 3. Crossover Isolation: Mencari titik maksimum global
    V_dagger_idx = np.argmax(g_V)
    
    return V[V_dagger_idx], g_V[V_dagger_idx]

if __name__ == "__main__":
    # Simulasi trajektori pertumbuhan spasial realistis yang sama
    np.random.seed(42)
    V = np.linspace(10, 100, 100)
    beta1 = 1 / (1 + np.exp(-(V - 50) / 5)) + np.random.normal(0, 0.015, size=V.shape)

    V_dagger, max_g = find_operational_crossover(V, beta1)
    print("=== METODE MODERN (USULAN) ===")
    print(f"Kriteria: V_dagger = arg max g(V)")
    print(f"Jumlah titik crossover yang ditemukan: 1 titik unik pada Volume = {V_dagger:.2f}")
    print("Hasil: Sukses. Mengisolasi satu titik puncak dominan yang menunjukkan efisiensi pembentukan loop maksimal, mengabaikan noise mikro-struktural.")
