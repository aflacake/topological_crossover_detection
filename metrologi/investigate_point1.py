import numpy as np

def compute_cubical_euler_characteristic(binary_grid):
    n_voxels = np.sum(binary_grid)
    if n_voxels == 0:
        return 0, 0, 0
    
    padded = np.pad(binary_grid, pad_width=1, mode='wrap')
    
    # Faces (C2)
    fx = np.sum(padded[:-1, :, :] & padded[1:, :, :])
    fy = np.sum(padded[:, :-1, :] & padded[:, 1:, :])
    fz = np.sum(padded[:, :, :-1] & padded[:, :, 1:])
    n_faces = fx + fy + fz
    
    # Edges (C1)
    e_xy = np.sum(padded[:-1, :-1, :] & padded[1:, :-1, :] & padded[:-1, 1:, :] & padded[1:, 1:, :])
    e_xz = np.sum(padded[:-1, :, :-1] & padded[1:, :, :-1] & padded[:-1, :, 1:] & padded[1:, :, 1:])
    e_yz = np.sum(padded[:, :-1, :-1] & padded[:, 1:, :-1] & padded[:, :-1, 1:] & padded[:, 1:, 1:])
    n_edges = e_xy + e_xz + e_yz
    
    # Nodes (C0)
    n_nodes = np.sum(
        padded[:-1, :-1, :-1] & padded[1:, :-1, :-1] & padded[:-1, 1:, :-1] & padded[1:, 1:, :-1] &
        padded[:-1, :-1, 1:]  & padded[1:, :-1, 1:]  & padded[:-1, 1:, 1:]  & padded[1:, 1:, 1:]
    )
    
    chi = n_voxels - n_faces + n_edges - n_nodes
    
    # Area
    diff_x = np.abs(np.diff(np.pad(binary_grid, ((1,1),(0,0),(0,0)), mode='wrap'), axis=0))
    diff_y = np.abs(np.diff(np.pad(binary_grid, ((0,0),(1,1),(0,0)), mode='wrap'), axis=1))
    diff_z = np.abs(np.diff(np.pad(binary_grid, ((0,0),(0,0),(1,1)), mode='wrap'), axis=2))
    area = np.sum(diff_x) + np.sum(diff_y) + np.sum(diff_z)
    
    return n_voxels, area, chi

def investigate_point1_clean(L=32, num_samples=5, num_steps=120):
    print(f"===========================================================")
    print(f"  INVESTIGASI POIN 1 (MURNI DIFERENSIAL LOKAL): L = {L}    ")
    print(f"===========================================================\n")
    
    e_star_list = []
    p_crit_list = []
    
    for s in range(num_samples):
        # Set seed opsional jika ingin variasi per sampel tetapi reproduksibel
        np.random.seed(s + 42)
        grid_prob = np.random.rand(L, L, L)
        p_values = np.linspace(0.28, 0.35, num_steps) # Rampitkan rentang p di sekitar 0.311
        
        volumes, areas, eulers = [], [], []
        
        for p in p_values:
            binary_grid = grid_prob <= p
            V, A, chi = compute_cubical_euler_characteristic(binary_grid)
            if V > 0 and A > 0:
                volumes.append(V)
                areas.append(A)
                eulers.append(chi)
                
        volumes = np.array(volumes)
        areas = np.array(areas)
        eulers = np.array(eulers)
        
        zero_crossings = np.where(np.diff(np.signbit(eulers)))[0]
        if len(zero_crossings) == 0:
            continue
            
        idx = zero_crossings[len(zero_crossings) // 2]
        
        # 1. Interpolasi p* saat Chi = 0
        p1, p2 = p_values[idx], p_values[idx+1]
        chi1, chi2 = eulers[idx], eulers[idx+1]
        p_crit = p1 - chi1 * (p2 - p1) / (chi2 - chi1)
        p_crit_list.append(p_crit)
        
        # 2. TURUNAN DIFERENSIAL MURNI d(ln A) / d(ln V) pada interval terkecil [idx, idx+1]
        ln_V1, ln_V2 = np.log(volumes[idx]), np.log(volumes[idx+1])
        ln_A1, ln_A2 = np.log(areas[idx]), np.log(areas[idx+1])
        
        d_ln_V = ln_V2 - ln_V1
        d_ln_A = ln_A2 - ln_A1
        
        E_star_pure = d_ln_A / d_ln_V
        e_star_list.append(E_star_pure)
        
        print(f"SAMPLE #{s+1}")
        print(f"  Boundary Chi : [{chi1:3d}] -> [{chi2:3d}] (p* = {p_crit:.6f})")
        print(f"  d(ln V)      : {d_ln_V:.6f} | d(ln A): {d_ln_A:.6f}")
        print(f"  Elastisitas E*: {E_star_pure:.6f}\n")

    print("===========================================================")
    print("                    HASIL PEMBERSIHAN                      ")
    print("===========================================================")
    print(f"Rata-rata p* : {np.mean(p_crit_list):.6f} ± {np.std(p_crit_list):.6f}")
    print(f"Rata-rata E* : {np.mean(e_star_list):.6f} ± {np.std(e_star_list):.6f}")
    print("===========================================================\n")

if __name__ == "__main__":
    investigate_point1_clean(L=32, num_samples=5, num_steps=120)