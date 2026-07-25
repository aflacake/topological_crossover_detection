import numpy as np
from scipy.stats import linregress
import time

def compute_cubical_metrics_3d(grid):
    """
    Menghitung Volume (V), Luas Permukaan (A), dan Karakteristik Euler (chi)
    pada komplek kubikal 3D tanpa pembatasan batas (zero-padded).
    """
    if not np.any(grid):
        return 0, 0, 0
    
    # Pad 1 lapis di sekeliling kisi untuk menangani batas secara kontinu
    g = np.pad(grid, 1, mode='constant', constant_values=False)
    
    # Volume (N3): Jumlah sel 3D aktif
    V = np.sum(g)
    
    # Luas Permukaan (A): Jumlah muka 2D terekspos (antara True & False)
    fx_exp = g[:-1, :, :] ^ g[1:, :, :]
    fy_exp = g[:, :-1, :] ^ g[:, 1:, :]
    fz_exp = g[:, :, :-1] ^ g[:, :, 1:]
    A = np.sum(fx_exp) + np.sum(fy_exp) + np.sum(fz_exp)
    
    # Sel-2 (N2): Semua muka aktif dalam komplek
    fx = g[:-1, :, :] | g[1:, :, :]
    fy = g[:, :-1, :] | g[:, 1:, :]
    fz = g[:, :, :-1] | g[:, :, 1:]
    N2 = np.sum(fx) + np.sum(fy) + np.sum(fz)
    
    # Sel-1 (N1): Semua rusuk aktif dalam komplek
    ex = g[:, :-1, :-1] | g[:, :-1, 1:] | g[:, 1:, :-1] | g[:, 1:, 1:]
    ey = g[:-1, :, :-1] | g[:-1, :, 1:] | g[1:, :, :-1] | g[1:, :, 1:]
    ez = g[:-1, :-1, :] | g[:-1, 1:, :] | g[1:, :-1, :] | g[1:, 1:, :]
    N1 = np.sum(ex) + np.sum(ey) + np.sum(ez)
    
    # Sel-0 (N0): Semua simpul/puncak aktif
    v0 = (g[:-1, :-1, :-1] | g[:-1, :-1, 1:] | g[:-1, 1:, :-1] | g[:-1, 1:, 1:] |
          g[1:, :-1, :-1] | g[1:, :-1, 1:] | g[1:, 1:, :-1] | g[1:, 1:, 1:])
    N0 = np.sum(v0)
    
    # Karakteristik Euler (chi) = N0 - N1 + N2 - N3
    chi = N0 - N1 + N2 - V
    return V, A, chi


def run_high_precision_percolation(
    L_list=[64, 96, 128],
    p_min=0.30,
    p_max=0.33,
    num_steps=500,
    num_samples=10,
    window_size=7
):
    """
    Eksperimen Perkolasi Presisi Tinggi dengan:
    1. Perhalusan sampling p (500-1000 langkah)
    2. Fitting Regresi Lokal W-titik (5-9 titik) di sekitar chi = 0
    """
    print("=" * 75)
    print("      SIMULASI PERKOLASI PRESISI TINGGI: PENGUNCIAN FIXED POINT E*")
    print("=" * 75)
    print(f" * Rentang Sampling p   : [{p_min:.3f}, {p_max:.3f}] ({num_steps} langkah)")
    print(f" * Jendela Regresi Lokal: {window_size} titik di sekitar chi = 0")
    print(f" * Jumlah Sampel / L    : {num_samples} realisasi\n")

    p_values = np.linspace(p_min, p_max, num_steps)
    results = {}

    for L in L_list:
        t0 = time.time()
        print(f"[+] Memproses Kisi L = {L} ...")
        
        p_crits = []
        e_stars = []
        r2_scores = []

        for s in range(num_samples):
            # Seed unik per sampel agar independen
            np.random.seed(s + 1000)
            grid_prob = np.random.rand(L, L, L)

            p_valid, v_list, a_list, chi_list = [], [], [], []

            # 1. SAMPLING BERUNTUT (Mencegah Index Mismatch)
            for p in p_values:
                grid = (grid_prob <= p)
                V, A, chi = compute_cubical_metrics_3d(grid)
                if V > 0 and A > 0:
                    p_valid.append(p)
                    v_list.append(V)
                    a_list.append(A)
                    chi_list.append(chi)

            p_arr = np.array(p_valid)
            V_arr = np.array(v_list)
            A_arr = np.array(a_list)
            chi_arr = np.array(chi_list)

            half_w = window_size // 2
            if len(chi_arr) >= window_size:
                # 2. CARI TITIK AKAR TOPOLOGI (chi mendekati 0)
                abs_chi = np.abs(chi_arr)
                raw_idx = np.argmin(abs_chi)
                
                # Batasi indeks agar window_size tidak melompati batas array
                idx = int(np.clip(raw_idx, half_w, len(chi_arr) - half_w - 1))

                # Interpolasi Linier Presisi untuk p* pada chi = 0
                chi1, chi2 = chi_arr[idx], chi_arr[idx + 1]
                p1, p2 = p_arr[idx], p_arr[idx + 1]
                
                if abs(chi2 - chi1) > 1e-12:
                    p_crit = p1 - chi1 * (p2 - p1) / (chi2 - chi1)
                else:
                    p_crit = p1
                p_crits.append(p_crit)

                # 3. REGRESI LINIER LOKAL (W-titik)
                sub_start = idx - half_w
                sub_end = idx + half_w + 1

                sub_ln_V = np.log(V_arr[sub_start:sub_end])
                sub_ln_A = np.log(A_arr[sub_start:sub_end])

                slope, intercept, r_val, p_val, std_err = linregress(sub_ln_V, sub_ln_A)
                
                if not np.isnan(slope):
                    e_stars.append(slope)
                    r2_scores.append(r_val ** 2)

        elapsed = time.time() - t0
        mean_p = np.mean(p_crits)
        mean_E = np.mean(e_stars)
        std_E = np.std(e_stars)
        mean_r2 = np.mean(r2_scores)

        results[L] = {
            'p_crit': mean_p,
            'E_star': mean_E,
            'std_E': std_E,
            'r2': mean_r2
        }

        print(f"    -> Done in {elapsed:.2f}s | Mean p* = {mean_p:.6f} | Mean E* = {mean_E:.6f} | STD(E*) = {std_E:.6f} | Mean R^2 = {mean_r2:.5f}")

    # RANGKUMAN REKAPITULASI
    print("\n" + "=" * 75)
    print("                 REKAPITULASI HASIL SIMULASI PRESISI")
    print("=" * 75)
    print(f"{'Skala Kisi (L)':^15} | {'p* Kritis':^12} | {'Rerata E*':^12} | {'STD (Noise)':^12} | {'Selisih dari 0.5':^15}")
    print("-" * 75)
    for L, res in results.items():
        diff = abs(res['E_star'] - 0.5)
        print(f"{L:^15d} | {res['p_crit']:^12.6f} | {res['E_star']:^12.6f} | {res['std_E']:^12.6f} | {diff:^15.6f}")
    print("=" * 75)

    return results

if __name__ == "__main__":
    # Jalankan simulasi
    run_high_precision_percolation(
        L_list=[64, 96],       # Tambahkan 128 jika memori & waktu mencukupi
        num_steps=500,         # 500 - 1000 langkah
        num_samples=5,         # Jumlah sampel per ukuran kisi
        window_size=7          # 7 titik regresi lokal (3 kiri, 1 tengah, 3 kanan)
    )