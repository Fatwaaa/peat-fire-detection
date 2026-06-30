"""
scripts/predict.py (Versi Acak)

Membaca model U-Net dan melakukan prediksi pada area acak (random patch)
dari citra satelit untuk melihat variasi output yang berbeda-beda.
"""

import os
import sys
import glob
import random
import numpy as np
import torch
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path

# Setup direktori
PROCESSED_SENTINEL_DIR = os.path.join("data", "processed", "sentinel2")
MODEL_PATH = os.path.join("models", "best_unet_peatfire.pth")
OUTPUT_DIR = os.path.join("data", "output_predictions")

# Hubungkan ke arsitektur model dari skrip training
sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.train_unet import SimpleUNet

def predict_random_patch():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleUNet(in_channels=1, out_channels=1).to(device)
    
    if not os.path.exists(MODEL_PATH):
        print(f"File model {MODEL_PATH} tidak ditemukan. Jalankan training dulu.")
        return
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("[*] Model AI Berhasil Dipanggil ke Memori.")

    # 2. Ambil citra NBR
    nbr_files = glob.glob(os.path.join(PROCESSED_SENTINEL_DIR, "*_NBR.tif"))
    if not nbr_files:
        print("Tidak ada citra NBR untuk diuji.")
        return
    
    sample_image = nbr_files[0]
    print(f"[+] Membaca citra satelit: {os.path.basename(sample_image)}")

    # 3. Ambil porsi gambar SECARA ACAK (Random Patch 512x512)
    patch_size = 512
    with rasterio.open(sample_image) as src:
        height, width = src.shape
        
        # Batasi koordinat agar potongan tidak keluar dari batas gambar
        max_y = height - patch_size
        max_x = width - patch_size
        
        # Pilih koordinat X dan Y secara acak
        start_y = random.randint(0, max_y)
        start_x = random.randint(0, max_x)
        
        print(f"[+] Mengambil potongan acak pada koordinat piksel: X={start_x}, Y={start_y}")
        window = rasterio.windows.Window(start_x, start_y, patch_size, patch_size)
        
        nbr_patch = src.read(1, window=window).astype(np.float32)
        nbr_norm = (nbr_patch + 1.0) / 2.0

    # 4. Prediksi via Model
    input_tensor = torch.from_numpy(nbr_norm).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(input_tensor)
        prediction = prediction.squeeze().cpu().numpy()

    # 5. Threshold biner (Probabilitas > 0.5)
    binary_prediction = (prediction > 0.5).astype(np.uint8)

    # 6. Visualisasikan Hasil Akhir
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.title(f"NBR Input Satelit (Potongan Acak)")
    plt.imshow(nbr_patch, cmap="brg")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("Hasil Segmentasi Deteksi Kebakaran AI")
    plt.imshow(binary_prediction, cmap="gray")
    plt.colorbar()

    output_plot_path = os.path.join(OUTPUT_DIR, "hasil_deteksi_kebakaran_acak.png")
    plt.savefig(output_plot_path, bbox_inches='tight')
    plt.close()
    
    print(f"[SUKSES] Visualisasi baru berhasil disimpan di: {output_plot_path}")
    print("[Tips] Jalankan skrip ini lagi untuk mendapatkan potongan wilayah Riau yang berbeda!\n")

if __name__ == "__main__":
    predict_random_patch()