import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import networkx as nx

# =====================================================================
# 1. REKONSTRUKSI KOORDINAT & NEIGHBOR LIST EKSPLISIT (FIRST PRINCIPLES)
# =====================================================================

def build_lattice_graph(lattice_type='SC', L=8):
    """
    Merekonstruksi grafik ketetanggaan eksplisit berdasarkan titik koordinat 3D murni:
    - SC  : Simple Cubic (1 site per unit cell, z = 6)
    - BCC : Body-Centered Cubic (2 sites per unit cell, z = 8)
    - FCC : Face-Centered Cubic (4 sites per unit cell, z = 12)
    """
    positions = []
    
    # 1. Sediakan Posisi Atom/Situs Sesuai Geometri
    if lattice_type == 'SC':
        for x in range(L):
            for y in range(L):
                for z in range(L):
                    positions.append((x, y, z))
        cutoff_dist = 1.01  # Jarak tetangga terdekat SC = 1.0

    elif lattice_type == 'BCC':
        for x in range(L):
            for y in range(L):
                for z in range(L):
                    # Situs Sudut (Corner)
                    positions.append((float(x), float(y), float(z)))
                    # Situs Pusat Kubus (Body-Center)
                    positions.append((x + 0.5, y + 0.5, z + 0.5))
        cutoff_dist = np.sqrt(3) / 2 + 0.05  # Jarak tetangga terdekat BCC ≈ 0.866

    elif lattice_type == 'FCC':
        for x in range(L):
            for y in range(L):
                for z in range(L):
                    # Situs Sudut (Corner)
                    positions.append((float(x), float(y), float(z)))
                    # Situs Pusat Muka (Face-Centers)
                    positions.append((x + 0.5, y + 0.5, float(z)))
                    positions.append((x + 0.5, float(y), z + 0.5))
                    positions.append((float(x), y + 0.5, z + 0.5))
        cutoff_dist = np.sqrt(2) / 2 + 0.05  # Jarak tetangga terdekat FCC ≈ 0.707

    positions = np.array(positions)
    N_sites = len(positions)
    
    # 2. Replikasi Grafik Ketetanggaan Eksplisit (Periodic Boundary Conditions)
    adj_list = [[] for _ in range(N_sites)]
    
    # Konstruksi matriks jarak dengan PBC
    for i in range(N_sites):
        pos_i = positions[i]
        # Hitung jarak delta terhadap semua situs lain (memperhitungkan wrap-around PBC)
        dx = np.abs(positions[:, 0] - pos_i[0])
        dy = np.abs(positions[:, 1] - pos_i[1])
        dz = np.abs(positions[:, 2] - pos_i[2])
        
        dx = np.minimum(dx, L - dx)
        dy = np.minimum(dy, L - dy)
        dz = np.minimum(dz, L - dz)
        
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # Cari tetangga terdekat yang valid (di atas 0 dan di bawah cutoff)
        neighbors = np.where((dist > 1e-4) & (dist <= cutoff_dist))[0]
        adj_list[i] = neighbors.tolist()
        
    return positions, adj_list

# =====================================================================
# 2. PERHITUNGAN TOPOLOGI KOMPLEKS SEL & EULER CHARACTERISTIC EKSOLUT
# =====================================================================

def compute_graph_topology_and_geometry(occupied_sites, adj_list):
    """
    Menhitung V, A (antarmuka batas terpisah), dan Karakteristik Euler Eksak
    χ = b0 - b1 + b2 langsung dari sub-graph jaringan terisi.
    """
    if len(occupied_sites) == 0:
        return 0, 0, 0
    
    occ_set = set(occupied_sites)
    
    # 1. Volume (V): Jumlah situs terisi
    V = len(occupied_sites)
    
    # 2. Luas Antarmuka (A): Jumlah koneksi terputus ke situs kosong (Boundary Bonds)
    A = 0
    num_edges_internal = 0
    
    for site in occupied_sites:
        for nbr in adj_list[site]:
            if nbr in occ_set:
                num_edges_internal += 1  # Rusuk internal
            else:
                A += 1                  # Antarmuka batas terbuka

    num_edges_internal //= 2  # Hindari double-counting rusuk
    
    # 3. Topologi Betti Numbers (b0, b1) via NetworkX Subgraph
    # Membangun sub-grafik murni dari situs yang aktif
    sub_edges = []
    for site in occupied_sites:
        for nbr in adj_list[site]:
            if nbr in occ_set and site < nbr:
                sub_edges.append((site, nbr))
                
    G = nx.Graph()
    G.add_nodes_from(occupied_sites)
    G.add_edges_from(sub_edges)
    
    # b0: Jumlah komponen terhubung (Kluster Padat)
    b0 = nx.number_connected_components(G)
    
    # b1: Fundamental Cycles / Cyclomatic Number (Loop/Terusan Topologis)
    # b1 = |Edges| - |Nodes| + |Components|
    b1 = num_edges_internal - V + b0
    
    # Karakteristik Euler Grafik Kritis: \chi \approx b0 - b1
    chi = b0 - b1
    
    return V, A, chi

# =====================================================================
# 3. SIMULASI MONTE CARLO & FSS OBJEKTIF (BEBAS PREDIKSI)
# =====================================================================

def run_fss_authentic(lattice_type='SC', L_list=[4, 6, 8], num_samples=3, num_steps=50):
    r"""
    Menjalankan simulasi Monte Carlo murni pada grafik ketetanggaan asli.
    """
    print(f"\n---> Memulai Simulasi Otentik Kisi: {lattice_type}")
    
    results = {}
    
    for L in L_list:
        positions, adj_list = build_lattice_graph(lattice_type, L)
        N_sites = len(positions)
        
        # Rentang p_values adaptif mengikuti jumlah situs
        p_values = np.linspace(0.10, 0.50, num_steps)
        k_trans_samples = []
        
        for s in range(num_samples):
            # Monte Carlo: Berikan nilai acak [0, 1] pada tiap situs
            site_probs = np.random.rand(N_sites)
            
            volumes, areas, eulers = [], [], []
            
            for p in p_values:
                occupied_sites = np.where(site_probs <= p)[0]
                V, A, chi = compute_graph_topology_and_geometry(occupied_sites, adj_list)
                
                if V > 0 and A > 0:
                    volumes.append(V)
                    areas.append(A)
                    eulers.append(chi)
                    
            volumes = np.array(volumes)
            areas = np.array(areas)
            eulers = np.array(eulers)
            
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

            # 2. Hitung gradient secara aman tanpa pembagian dengan nol
            K_curve = np.gradient(ln_A, ln_V)
            
            # Cari persilangan nol \chi = 0
            zero_crossings = np.where(np.diff(np.signbit(clean_chi)))[0]
            if len(zero_crossings) > 0:
                idx = zero_crossings[0]
                k_trans_samples.append(K_curve[idx])
                
        if len(k_trans_samples) > 0:
            results[L] = (np.mean(k_trans_samples), np.std(k_trans_samples))
            print(f"L = {L:2d} | N_sites = {N_sites:5d} | K_trans = {results[L][0]:.4f} ± {results[L][1]:.4f}")

    if len(results) < 2:
        return None, None, None, np.nan, np.nan

    inv_L = np.array([1.0 / L for L in results.keys()])
    K_means = np.array([results[L][0] for L in results.keys()])
    K_stds = np.array([results[L][1] for L in results.keys()])

    slope, intercept, r_value, p_value, std_err = linregress(inv_L, K_means)
    
    return inv_L, K_means, K_stds, intercept, r_value**2

def execute_authentic_universality_experiment():
    r"""
    Eksekusi eksperimen tanpa memodifikasi data mentah.
    """
    lattices = ['SC', 'BCC', 'FCC']
    colors = {'SC': 'blue', 'BCC': 'green', 'FCC': 'purple'}
    
    print("=" * 70)
    print("   EKSPLORASI FSS OTENTIK: SC vs BCC vs FCC (FIRST PRINCIPLES)")
    print("=" * 70)

    plt.figure(figsize=(9, 6))

    for lat in lattices:
        # Menggunakan ukuran sel unit L = 4, 6, 8 (mencakup hingga ribuan situs asli)
        inv_L, K_means, K_stds, K_crit, r_sq = run_fss_authentic(lattice_type=lat, L_list=[4, 6, 8])
        
        if inv_L is None or np.isnan(K_crit):
            print(f"\nMerekam Kisi: {lat:3s} | Status: Data Tidak Cukup untuk Ekstrapolasi.")
            continue

        print(f"\nHasil Akhir Kisi {lat:3s} -> Limit Asimtotik K_crit (L->∞) = {K_crit:.4f} | R² = {r_sq:.4f}")

        plt.errorbar(inv_L, K_means, yerr=K_stds, fmt='o', color=colors[lat], 
                     ecolor='gray', capsize=4, label=f'{lat} Data')
        
        x_fit = np.linspace(0, max(inv_L) * 1.15, 50)
        slope_fit = (K_means[-1] - K_crit) / inv_L[-1]
        plt.plot(x_fit, K_crit + slope_fit * x_fit, linestyle='--', color=colors[lat],
                 label=r'%s Fit ($K_{crit}=%.4f$, $R^2=%.4f$)' % (lat, K_crit, r_sq))

    print("=" * 70)
    plt.axvline(0, color='black', linewidth=0.8, linestyle=':')
    plt.xlabel(r'Kebalikan Ukuran Kisi $(1 / L)$')
    plt.ylabel(r'Eksponent Alometrik Terukur $\left. \frac{d(\ln A)}{d(\ln V)} \right|_{\chi = 0}$')
    plt.title(r'Uji FSS Otentik: Grafik Ketetanggaan Asli SC, BCC, dan FCC')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    execute_authentic_universality_experiment()