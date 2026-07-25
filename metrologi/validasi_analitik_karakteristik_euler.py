import numpy as np

def compute_euler_characteristic_exact(grid_3d):
    """
    Menghitung Karakteristik Euler (chi = N0 - N1 + N2 - N3) dari matriks 3D biner.
    Grid biner: 1 = voxel padat, 0 = kosong.
    """
    # Dapatkan koordinat semua voxel padat (N3)
    voxels = np.argwhere(grid_3d == 1)
    if len(voxels) == 0:
        return 0

    N3 = len(voxels)

    # Sets untuk menyimpan elemen unik (N0, N1, N2)
    vertices = set()
    edges = set()
    faces = set()

    # 8 simpul relatif dari sebuah voxel [0,0,0] ke [1,1,1]
    v_offsets = [
        (0,0,0), (1,0,0), (0,1,0), (1,1,0),
        (0,0,1), (1,0,1), (0,1,1), (1,1,1)
    ]

    # 12 rusuk relatif (pasangan simpul)
    e_offsets = [
        # Rusuk arah X
        ((0,0,0),(1,0,0)), ((0,1,0),(1,1,0)), ((0,0,1),(1,0,1)), ((0,1,1),(1,1,1)),
        # Rusuk arah Y
        ((0,0,0),(0,1,0)), ((1,0,0),(1,1,0)), ((0,0,1),(0,1,1)), ((1,0,1),(1,1,1)),
        # Rusuk arah Z
        ((0,0,0),(0,0,1)), ((1,0,0),(1,0,1)), ((0,1,0),(0,1,1)), ((1,1,0),(1,1,1))
    ]

    # 6 muka relatif (4 simpul terurut)
    f_offsets = [
        # Muka XY (Z=0 dan Z=1)
        ((0,0,0),(1,0,0),(1,1,0),(0,1,0)), ((0,0,1),(1,0,1),(1,1,1),(0,1,1)),
        # Muka XZ (Y=0 dan Y=1)
        ((0,0,0),(1,0,0),(1,0,1),(0,0,1)), ((0,1,0),(1,1,0),(1,1,1),(0,1,1)),
        # Muka YZ (X=0 dan X=1)
        ((0,0,0),(0,1,0),(0,1,1),(0,0,1)), ((1,0,0),(1,1,0),(1,1,1),(1,0,1))
    ]

    for vx, vy, vz in voxels:
        # 1. Catat Vertices (N0)
        for dx, dy, dz in v_offsets:
            vertices.add((vx + dx, vy + dy, vz + dz))

        # 2. Catat Edges (N1)
        for (x1,y1,z1), (x2,y2,z2) in e_offsets:
            p1 = (vx + x1, vy + y1, vz + z1)
            p2 = (vx + x2, vy + y2, vz + z2)
            # Urutkan pasangan titik agar unik
            edge = tuple(sorted([p1, p2]))
            edges.add(edge)

        # 3. Catat Faces (N2)
        for f in f_offsets:
            face_verts = tuple(sorted([(vx + dx, vy + dy, vz + dz) for dx, dy, dz in f]))
            faces.add(face_verts)

    N0 = len(vertices)
    N1 = len(edges)
    N2 = len(faces)

    chi = N0 - N1 + N2 - N3
    return chi, N0, N1, N2, N3


# =========================================================================
# EKSEKUSI SUITE UJI VALIDASI GEOMETRIS
# =========================================================================

def run_validation_suite():
    tests = {}

    # 1. Single Voxel (1x1x1)
    g1 = np.zeros((3,3,3), dtype=int)
    g1[1,1,1] = 1
    tests["1. Single Voxel"] = (g1, 1)

    # 2. Solid Cube (2x2x2 = 8 voxel)
    g2 = np.zeros((4,4,4), dtype=int)
    g2[1:3, 1:3, 1:3] = 1
    tests["2. Solid Cube (2x2x2)"] = (g2, 1)

    # 3. Two Adjacent Voxels (Sharing 1 Face)
    g3 = np.zeros((4,3,3), dtype=int)
    g3[1,1,1] = 1
    g3[2,1,1] = 1
    tests["3. Two Adjacent Voxels"] = (g3, 1)

    # 4. Two Disjoint Voxels (Separated)
    g4 = np.zeros((5,3,3), dtype=int)
    g4[1,1,1] = 1
    g4[3,1,1] = 1
    tests["4. Two Disjoint Voxels"] = (g4, 2)

    # 5. Digital Torus / Ring 3D (3x3 dengan lubang di tengah)
    g5 = np.zeros((5,5,3), dtype=int)
    # Buat ring 3x3 di bidang XY tebal 1 unit di Z
    g5[1:4, 1:4, 1] = 1
    g5[2, 2, 1] = 0  # Lubang di tengah
    tests["5. Digital Torus (Hole=1)"] = (g5, 0)

    print(f"{'Test Case':<28} | {'N0':<4} {'N1':<4} {'N2':<4} {'N3':<4} | {'Chi Hitung':<10} | {'Chi Teori':<10} | {'Status'}")
    print("-" * 80)

    all_passed = True
    for name, (grid, expected_chi) in tests.items():
        chi, n0, n1, n2, n3 = compute_euler_characteristic_exact(grid)
        passed = (chi == expected_chi)
        if not passed: all_passed = False
        status = "PASSED [OK]" if passed else "FAILED [X]"
        print(f"{name:<28} | {n0:<4} {n1:<4} {n2:<4} {n3:<4} | {chi:<10} | {expected_chi:<10} | {status}")

    print("-" * 80)
    if all_passed:
        print("HASIL VALIDASI: SEMUA TEST-CASE LOLOS EXAK! Formulasi chi valid secara matematis.")
    else:
        print("HASIL VALIDASI: ADA TEST-CASE YANG GAGAL! Periksa kembali definisi sel.")

if __name__ == "__main__":
    run_validation_suite()