"""
scripts/preprocess_sentinel2.py

Mengekstrak file .zip Sentinel-2, membaca Band 8 (NIR) dan Band 12 (SWIR2),
menghitung Normalized Burn Ratio (NBR), dan menyimpan hasilnya ke data/processed.
"""

import os
import sys
import zipfile
import glob
import numpy as np
import rasterio
from pathlib import Path

# Hubungkan ke config root proyek
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import RAW_SENTINEL2_DIR

PROCESSED_DIR = os.path.join("data", "processed", "sentinel2")

def extract_zip(zip_path, extract_to):
    """Mengekstrak file zip jika folder .SAFE belum ada."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Cari nama folder utama di dalam zip (biasanya berakhiran .SAFE)
        top_folder = zip_ref.namelist()[0].split('/')[0]
        target_dir = os.path.join(extract_to, top_folder)
        
        if os.path.exists(target_dir):
            print(f"  [skip] Sudah diekstrak: {top_folder}")
            return target_dir
            
        print(f"  Mengekstrak {os.path.basename(zip_path)}...")
        zip_ref.extractall(extract_to)
        return target_dir

from rasterio.enums import Resampling

def calc_nbr(band8_path, band12_path, out_path):
    """Membaca band NIR dan SWIR2, melakukan resampling SWIR2 ke 10m, menghitung NBR."""
    print("  Menghitung indeks NBR (dengan resampling)...")
    with rasterio.open(band8_path) as b8, rasterio.open(band12_path) as b12:
        # 1. Baca Band 8 (NIR) resolusi asli 10m
        nir = b8.read(1).astype('float32')
        
        # 2. Baca Band 12 (SWIR2) dan paksa lakukan resampling ke dimensi Band 8 (10m)
        swir2 = b12.read(
            1,
            out_shape=(b8.height, b8.width),
            resampling=Resampling.bilinear
        ).astype('float32')
        
        # 3. Penanganan pembagian dengan nol (zero division)
        denominator = nir + swir2
        denominator[denominator == 0] = 1e-5
        
        # 4. Hitung NBR (sekarang dimensi array sudah sama-sama 10980 x 10980)
        nbr = (nir - swir2) / denominator
        
        # 5. Salin metadata dari Band 8 (karena output kita sekarang beresolusi 10m)
        meta = b8.meta.copy()
        meta.update({
            "driver": "GTiff",
            "dtype": "float32",
            "count": 1,
            "nodata": -9999
        })
        
        # Simpan hasil kalkulasi matrix NBR ke folder processed
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(nbr, 1)
            
    print(f"  [sukses] Tersimpan di: {out_path}")

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    zip_files = glob.glob(os.path.join(RAW_SENTINEL2_DIR, "*.zip"))
    
    if not zip_files:
        print(f"Tidak ada file .zip ditemukan di {RAW_SENTINEL2_DIR}. Silakan download data terlebih dahulu.")
        return

    print(f"Menemukan {len(zip_files)} file zip untuk diproses.\n")
    
    for zip_path in zip_files:
        scene_name = Path(zip_path).stem
        print(f"Memproses: {scene_name}")
        
        # 1. Ekstrak file zip langsung ke direktori raw sementara
        safe_dir = extract_zip(zip_path, RAW_SENTINEL2_DIR)
        
        # 2. Cari lokasi file Band 8 (10m) dan Band 12 (20m) di dalam struktur folder .SAFE
        # Sentinel-2 L2A menyimpan band resolusi penuh di dalam folder GRANULE/.../IMG_DATA/R10m atau R20m
        b8_match = glob.glob(os.path.join(safe_dir, "GRANULE", "*", "IMG_DATA", "R10m", "*_B08_10m.jp2"))
        b12_match = glob.glob(os.path.join(safe_dir, "GRANULE", "*", "IMG_DATA", "R20m", "*_B12_20m.jp2"))
        
        if not b8_match or not b12_match:
            print("  [Error] File Band 8 atau Band 12 tidak ditemukan dalam folder .SAFE")
            continue
            
        out_nbr_tiff = os.path.join(PROCESSED_DIR, f"{scene_name}_NBR.tif")
        
        # 3. Hitung dan simpan NBR
        calc_nbr(b8_match[0], b12_match[0], out_nbr_tiff)
        print("-" * 50)

if __name__ == "__main__":
    main()