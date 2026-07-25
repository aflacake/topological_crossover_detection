import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Set seed agar hasil dokumentasi 100% REPRODUSIBEL
np.random.seed(42)

def build_lattice_cell_complex(lattice_type='SC', L=6):
    """
    Merekonstruksi Situs (C0), Rusuk (C1), dan Muka/Plaquette (C2) 
    secara eksplisit sesuai geometri kristal murni.
    """
    positions = []
    if lattice_type == 'SC':
        for x in range(L):
            for y in range(L):
                for z in range(L):
                    positions.append((float(x), float(y), float(z)))
        cutoff_e = 1.01
    elif lattice_type == 'BCC':
        for x in range(L):
            for y in range(L):
                for z in range(L):
                    positions.append((float(x), float(y), float(z)))
                    positions.append((x + 0.5, y + 0.5, z + 0.5))
        cutoff_e = np.sqrt(3)/2 + 0.05
    elif lattice_type == 'FCC':
        for x in range(L):
            for y in range(L):
                for z in range(L):
                    positions.append((float(x), float(y), float(z)))
                    positions.append((x + 0.5, y + 0.5, float(z)))
                    positions.append((x + 0.5, float(y), z + 0.5))
                    positions.append((float(x), y + 0.5, z + 0.5))
        cutoff_e = np.sqrt(2)/2 + 0.05

    positions = np.array(positions)
    N = len(positions)
    
    # 1. Edge List (C1)
    adj_list = [[] for _ in range(N)]
    edges = set()
    for i in range(N):
        pos_i = positions[i]
        dx = np.abs(positions[:, 0] - pos_i[0])
        dy = np.abs(positions[:, 1] - pos_i[1])
        dz = np.abs(positions[:, 2] - pos_i[2])
        dx = np.minimum(dx, L - dx)
        dy = np.minimum(dy, L - dy)
        dz = np.minimum(dz, L - dz)
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        nbrs = np.where((dist > 1e-4) & (dist <= cutoff_e))[0]
        adj_list[i] = nbrs.tolist()
        for nbr in nbrs:
            if i < nbr:
                edges.add((i, nbr))

    # 2. Face / Plaquette List (C2) - Deteksi Loop Tertutup Terkecil (Segitiga / Segiempat)
    faces = []
    if lattice_type in ['BCC', 'FCC']:
        # Loop Segitiga (3 situs saling terhubung)
        for u in range(N):
            nbrs_u = set(adj_list[u])
            for v in adj_list[u]:
                if u < v:
                    common = nbrs_u.intersection(adj_list[v])
                    for w in common:
                        if v < w:
                            faces.append((u, v, w))
    else: # SC: Loop Segiempat (Square Plaquette)
        for u, v in edges:
            nbrs_u = set(adj_list[u])
            nbrs_v = set(adj_list[v])
            for w in nbrs_u:
                if w != v:
                    for x in nbrs_v:
                        if x != u and x in adj_list[w]:
                            sq = tuple(sorted([u, v, w, x]))
                            if sq not in faces:
                                faces.append(sq)

    return N, adj_list, edges, faces

def run_fss_exact_cell_complex(lattices=['SC', 'BCC', 'FCC'], L_list=[4, 6, 8], num_samples=30):
    colors = {'SC': 'blue', 'BCC': 'green', 'FCC': 'purple'}
    
    print("=" * 75)
    print("   FSS EXACT CELL-COMPLEX (C0 - C1 + C2): SC vs BCC vs FCC")
    print("=" * 75)

    plt.figure(figsize=(10, 6))

    for lat in lattices:
        print(f"\n---> Memulai Simulasi Otentik Kisi: {lat}")
        lattice_data = {}

        for L in L_list:
            N_total, adj_list, edges, faces = build_lattice_cell_complex(lat, L)
            # Fokus rentang probabilitas di sekitar p_c masing-masing kisi
            p_values = np.linspace(0.10, 0.60, 100)
            k_trans_samples = []

            for s in range(num_samples):
                site_probs = np.random.rand(N_total)

                volumes, areas, eulers = [], [], []

                for p in p_values:
                    occ_mask = site_probs <= p
                    occ_set = set(np.where(occ_mask)[0])
                    
                    V = len(occ_set)
                    if V < 4:
                        continue

                    # Count Edges (C1) & Boundary Area (A)
                    E_count = 0
                    A_count = 0
                    for u, v in edges:
                        u_occ = u in occ_set
                        v_occ = v in occ_set
                        if u_occ and v_occ:
                            E_count += 1
                        elif u_occ or v_occ:
                            A_count += 1

                    # Count Faces (C2)
                    F_count = 0
                    if lat in ['BCC', 'FCC']:
                        for u, v, w in faces:
                            if u in occ_set and v in occ_set and w in occ_set:
                                F_count += 1
                    else: # SC
                        for u, v, w, x in faces:
                            if u in occ_set and v in occ_set and w in occ_set and x in occ_set:
                                F_count += 1

                    # Karakteristik Euler Kompleks Selular Eksak
                    chi = V - E_count + F_count

                    volumes.append(V)
                    areas.append(A_count)
                    eulers.append(chi)

                volumes = np.array(volumes)
                areas = np.array(areas)
                eulers = np.array(eulers)

                # Filter Duplikasi
                unique_mask = np.ones(len(volumes), dtype=bool)
                for i in range(1, len(volumes)):
                    if volumes[i] <= volumes[i-1] or areas[i] <= 0:
                        unique_mask[i] = False

                clean_V = volumes[unique_mask]
                clean_A = areas[unique_mask]
                clean_chi = eulers[unique_mask]

                if len(clean_V) < 5:
                    continue

                ln_V = np.log(clean_V)
                ln_A = np.log(clean_A)
                K_curve = np.gradient(ln_A, ln_V)

                # Deteksi Titik Involusi Topologi (\chi = 0)
                zero_crossings = np.where(np.diff(np.signbit(clean_chi)))[0]
                if len(zero_crossings) > 0:
                    idx = zero_crossings[0]
                    k_trans_samples.append(K_curve[idx])

            if len(k_trans_samples) > 0:
                mean_k = np.mean(k_trans_samples)
                std_k = np.std(k_trans_samples)
                lattice_data[L] = (mean_k, std_k)
                print(f"L = {L:2d} | N_sites = {N_total:5d} | K_trans = {mean_k:.4f} ± {std_k:.4f}")

        if len(lattice_data) >= 2:
            inv_L = np.array([1.0 / L for L in lattice_data.keys()])
            K_means = np.array([lattice_data[L][0] for L in lattice_data.keys()])
            K_stds = np.array([lattice_data[L][1] for L in lattice_data.keys()])

            slope, intercept, r_value, p_value, std_err = linregress(inv_L, K_means)
            r_sq = r_value**2

            print(f"Hasil Akhir Kisi {lat:3s} -> Limit Asimtotik K_crit (L->∞) = {intercept:.4f} | R² = {r_sq:.4f}")

            plt.errorbar(inv_L, K_means, yerr=K_stds, fmt='o', color=colors[lat], 
                         ecolor='gray', capsize=4, label=f'{lat} Data')
            
            x_fit = np.linspace(0, max(inv_L) * 1.15, 50)
            plt.plot(x_fit, intercept + slope * x_fit, linestyle='--', color=colors[lat],
                     label=r'%s Fit ($K_{crit}=%.4f$, $R^2=%.4f$)' % (lat, intercept, r_sq))

    print("=" * 75)
    plt.axvline(0, color='black', linewidth=0.8, linestyle=':')
    plt.xlabel(r'Kebalikan Ukuran Kisi $(1 / L)$')
    plt.ylabel(r'Eksponen Alometrik Terukur $\left. \frac{d(\ln A)}{d(\ln V)} \right|_{\chi = 0}$')
    plt.title(r'Uji FSS Exact Cell-Complex ($\chi = C_0 - C_1 + C_2$): SC, BCC, dan FCC')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_fss_exact_cell_complex(lattices=['SC', 'BCC', 'FCC'], L_list=[4, 6, 8], num_samples=30)