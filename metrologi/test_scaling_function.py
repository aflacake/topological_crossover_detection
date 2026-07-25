import numpy as np

def test_dynamic_scaling_function():
    # 1. Parameter Hasil Ekstrapolasi WLS Terkunci
    E_infinity = 0.501049
    slope = -0.749226
    
    # Data Observasi Log dari Simulasi Poin 2
    observasi_data = {
        20: {"E_mean": 0.444630, "std": 0.209941},
        32: {"E_mean": 0.503337, "std": 0.146087},
        48: {"E_mean": 0.480762, "std": 0.077730}
    }
    
    print("=========================================================================")
    print("      UJI PERSAMAAN PENSKALAAN DINAMIS E*(L) TANPA SPEKULASI            ")
    print("=========================================================================\n")
    print(f"Persamaan Model  : E*(L) = {E_infinity:.6f} + ({slope:.6f} / L)")
    print(f"Fixed Point L->inf: E* = {E_infinity:.6f}\n")
    
    print("-------------------------------------------------------------------------")
    print("  L  |  E* Observasi  |  E* Prediksi Model  |  Residual | Status (Err < STD)")
    print("-------------------------------------------------------------------------")
    
    residuals = []
    
    for L, data in observasi_data.items():
        E_obs = data["E_mean"]
        std_obs = data["std"]
        
        # Evaluasi Fungsi Penskalaan Dinamis
        E_pred = E_infinity + (slope / L)
        
        residual = abs(E_obs - E_pred)
        residuals.append(residual)
        
        # Validasi apakah perbedaan berada di bawah batas fluktuasi alami (STD)
        valid = residual < std_obs
        status = "PAS (Sesuai STD)" if valid else "DI LUAR TOLERANSI"
        
        print(f" {L:2d}  |    {E_obs:.6f}    |      {E_pred:.6f}       |  {residual:.6f}  | {status}")
        
    print("-------------------------------------------------------------------------")
    print(f"Rata-rata Residual (Absolute Error) : {np.mean(residuals):.6f}")
    print("=========================================================================\n")

    # 2. Pengujian Prediksi Ekstrapolasi untuk Kisi Skala Besar (L = 64, 128, 256)
    print("=========================================================================")
    print("      PREDIKSI ELASTISITAS E*(L) PADA KISI LARGER-SCALE (L -> inf)      ")
    print("=========================================================================")
    for L_large in [64, 128, 256, 512, 1024]:
        E_pred_large = E_infinity + (slope / L_large)
        deviasi_dari_fixed_point = abs(E_pred_large - E_infinity)
        print(f"  L = {L_large:4d}  |  E*(L) = {E_pred_large:.6f}  |  Deviasi dari E*inf: {deviasi_dari_fixed_point:.6f}")
    print("=========================================================================\n")

if __name__ == "__main__":
    test_dynamic_scaling_function()