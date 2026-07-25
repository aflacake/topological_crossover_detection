import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, binary_dilation, generate_binary_structure
from scipy.spatial.distance import cdist
from scipy.stats import linregress

def extract_empirical_p_contact(L=30, p=0.3116, num_realizations=5):
    """
    Mengekstrak P_contact(r) secara empiris dari kisi 3D
    dengan mengukur frekuensi penyatuan permukaan lokal pada jarak r.
    """
    print("===============================================================")
    print("  ESTIMASI EMPIRIS P_contact(r) DARI SIMULASI PERKOLASI 3D  ")
    print("===============================================================")

    # Structure 6-connectivity untuk label klaster
    struct_6 = generate_binary_structure(3, 1)

    r_bins = np.arange(1.0, 8.5, 0.5)
    total_pairs_at_r = np.zeros(len(r_bins) - 1)
    contact_events_at_r = np.zeros(len(r_bins) - 1)

    for realization in range(num_realizations):
        grid = (np.random.rand(L, L, L) < p).astype(int)
        labeled, num_clusters = label(grid, structure=struct_6)

        # Ambil klaster berukuran signifikan (V > 50)
        for cid in range(1, num_clusters + 1):
            mask = (labeled == cid)
            if np.sum(mask) < 50:
                continue

            # 1. Cari permukaan luar klaster
            dilated = binary_dilation(mask, structure=struct_6)
            boundary = dilated & (~mask) # Voksel kosong tepat di luar klaster
            boundary_coords = np.argwhere(boundary)

            if len(boundary_coords) < 10:
                continue

            # 2. Hitung jarak Euclidean antar semua pasangan titik boundary
            dist_matrix = cdist(boundary_coords, boundary_coords)

            # Abaikan tetangga sangat dekat (r < 1.5) untuk menghindari self-coupling trivial
            valid_mask = (dist_matrix >= 1.5) & (dist_matrix < 8.0)
            
            # 3. Uji apakah dua titik permukaan dapat menyatu (contact) jika diisi 1 voksel perantara
            # Penyatuan terjadi jika jarak Euclidean r <= sqrt(3) ~ 1.73 (tetangga kubik)
            contact_matrix = valid_mask & (dist_matrix <= np.sqrt(3))

            # 4. Akumulasi data ke dalam bin jarak r
            for i in range(len(r_bins) - 1):
                r_min, r_max = r_bins[i], r_bins[i+1]
                in_bin = valid_mask & (dist_matrix >= r_min) & (dist_matrix < r_max)
                
                n_pairs = np.sum(in_bin) // 2
                n_contacts = np.sum(contact_matrix & in_bin) // 2

                total_pairs_at_r[i] += n_pairs
                contact_events_at_r[i] += n_contacts

    # Hitung P_contact empiris = contacts / total_pairs
    p_contact_empirical = np.divide(
        contact_events_at_r, 
        total_pairs_at_r, 
        out=np.zeros_like(contact_events_at_r, dtype=float), 
        where=total_pairs_at_r > 0
    )

    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    
    # Filter bin yang memiliki sampel cukup
    valid_pts = total_pairs_at_r > 10
    r_valid = r_centers[valid_pts]
    p_valid = p_contact_empirical[valid_pts]

    # --- PLOTTING HASIL EKSPERIMEN ---
    plt.figure(figsize=(9, 5.5))
    plt.plot(r_valid, p_valid, 'o-', color='crimson', linewidth=2, markersize=7, label=r'Data Empiris $P_{\mathrm{contact}}(r)$')

    # Bandingkan dengan Fit Model Fenomenologis (Ansatz): P(r) = P0 * exp(-r / xi)
    if len(r_valid) > 2 and np.all(p_valid > 0):
        log_p = np.log(p_valid)
        slope, intercept, r_val, _, _ = linregress(r_valid, log_p)
        xi_fit = -1.0 / slope
        p0_fit = np.exp(intercept)

        r_smooth = np.linspace(min(r_valid), max(r_valid), 100)
        p_ansatz = p0_fit * np.exp(-r_smooth / xi_fit)
        
        plt.plot(r_smooth, p_ansatz, 'b--', label=rf'Fit Ansatz Eksponensial: $P_0 e^{{-r/\xi}}$ ($\xi \approx {xi_fit:.2f}, R^2={r_val**2:.3f}$)')

    plt.xlabel(r'Jarak Spasial Antar-Cabang $r$ (voxel)', fontsize=11, fontweight='bold')
    plt.ylabel(r'Probabilitas Self-Contact Empiris $P_{\mathrm{contact}}(r)$', fontsize=11, fontweight='bold')
    plt.title(r'Uji Disiplin Sains: Estimasi Empiris $P_{\mathrm{contact}}(r)$ tanpa Asumsi Awal', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

    print("\nHASIL EVALUASI EMPIRIS:")
    for r_val, p_val, n_sample in zip(r_valid, p_valid, total_pairs_at_r[valid_pts]):
        print(f"• Jarak r = {r_val:.2f} voxel | P_contact = {p_val:.4f} (diuji dari {int(n_sample)} pasangan)")

if __name__ == "__main__":
    extract_empirical_p_contact(L=32, p=0.3116, num_realizations=4)