import numpy as np
import matplotlib.pyplot as plt

def compute_cubical_euler_characteristic(binary_grid):
    n_voxels = np.sum(binary_grid)
    if n_voxels == 0:
        return 0, 0, 0
    
    padded = np.pad(binary_grid, pad_width=1, mode='wrap')
    
    # Faces (C2)
    fx = np.sum(padded[:-1, :, :] & padded[1:, :, :])
    fy = np.sum(padded[:, :-1, :] & padded[:, 1:, :])
    fz = np.sum(padded[:, :, :-1] & padded[:, :, 1:])
    
    # Edges (C1)
    e_xy = np.sum(padded[:-1, :-1, :] & padded[1:, :-1, :] & padded[:-1, 1:, :] & padded[1:, 1:, :])
    e_xz = np.sum(padded[:-1, :, :-1] & padded[1:, :, :-1] & padded[:-1, :, 1:] & padded[1:, :, 1:])
    e_yz = np.sum(padded[:, :-1, :-1] & padded[:, 1:, :-1] & padded[:, :-1, 1:] & padded[:, 1:, 1:])
    
    # Nodes (C0)
    n_nodes = np.sum(
        padded[:-1, :-1, :-1] & padded[1:, :-1, :-1] & padded[:-1, 1:, :-1] & padded[1:, 1:, :-1] &
        padded[:-1, :-1, 1:]  & padded[1:, :-1, 1:]  & padded[:-1, 1:, 1:]  & padded[1:, 1:, 1:]
    )
    
    chi = n_voxels - (fx + fy + fz) + (e_xy + e_xz + e_yz) - n_nodes
    
    diff_x = np.abs(np.diff(np.pad(binary_grid, ((1,1),(0,0),(0,0)), mode='wrap'), axis=0))
    diff_y = np.abs(np.diff(np.pad(binary_grid, ((0,0),(1,1),(0,0)), mode='wrap'), axis=1))
    diff_z = np.abs(np.diff(np.pad(binary_grid, ((0,0),(0,0),(1,1)), mode='wrap'), axis=2))
    area = np.sum(diff_x) + np.sum(diff_y) + np.sum(diff_z)
    
    return n_voxels, area, chi

def investigate_point2_scaling(L_list=[20, 32, 48], num_samples=6, num_steps=100):
    print("=====================================================================")
    print("  INVESTIGASI POIN 2: CONVERGENCE & DECAY TO FIXED POINT (L -> inf)  ")
    print("=====================================================================\n")
    
    summary_results = {}
    
    for L in L_list:
        print(f"[+] Memproses Skala Kisi L = {L:2d} ...")
        e_stars = []
        p_crits = []
        
        for s in range(num_samples):
            np.random.seed(s + 100)
            grid_prob = np.random.rand(L, L, L)
            p_values = np.linspace(0.28, 0.35, num_steps)
            
            volumes, areas, eulers = [], [], []
            for p in p_values:
                V, A, chi = compute_cubical_euler_characteristic(grid_prob <= p)
                if V > 0 and A > 0:
                    volumes.append(V)
                    areas.append(A)
                    eulers.append(chi)
                    
            volumes, areas, eulers = np.array(volumes), np.array(areas), np.array(eulers)
            zero_crossings = np.where(np.diff(np.signbit(eulers)))[0]
            
            if len(zero_crossings) == 0:
                continue
                
            idx = zero_crossings[len(zero_crossings) // 2]
            
            # Interpolasi p*
            p1, p2 = p_values[idx], p_values[idx+1]
            chi1, chi2 = eulers[idx], eulers[idx+1]
            p_crit = p1 - chi1 * (p2 - p1) / (chi2 - chi1)
            p_crits.append(p_crit)
            
            # Evaluasi E* Diferensial Lokal
            d_ln_V = np.log(volumes[idx+1]) - np.log(volumes[idx])
            d_ln_A = np.log(areas[idx+1]) - np.log(areas[idx])
            
            if d_ln_V > 0:
                E_star = d_ln_A / d_ln_V
                e_stars.append(E_star)
                
        mean_E = np.mean(e_stars)
        std_E = np.std(e_stars)
        summary_results[L] = (mean_E, std_E, np.mean(p_crits))
        
        print(f"    -> L = {L:2d} | Rerata E* = {mean_E:.6f} | Deviasi (Fluktuasi) STD = {std_E:.6f}")
        
    print("\n=====================================================================")
    print("                    ANALISIS PENSKALAAN (SUMMARY)                    ")
    print("=====================================================================")
    print(" Ukuran Kisi (L)  |   Rerata E*   |  Fluktuasi (STD)  |  Kerapatan p* ")
    print("---------------------------------------------------------------------")
    for L in L_list:
        m_E, s_E, m_p = summary_results[L]
        print(f"      L = {L:2d}       |   {m_E:.6f}    |     {s_E:.6f}     |   {m_p:.6f}")
    print("=====================================================================\n")

if __name__ == "__main__":
    investigate_point2_scaling(L_list=[20, 32, 48], num_samples=6, num_steps=100)