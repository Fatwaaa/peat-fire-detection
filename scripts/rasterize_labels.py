"""
scripts/rasterize_labels.py

Membaca data hotspot FIRMS (.csv) dan membakarnya (rasterize) menjadi gambar 
masker biner (.tif) yang sejajar persis dengan citra NBR Sentinel-2.
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import Point
from pathlib import Path

# Hubungkan ke config root proyek
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import RAW_FIRMS_DIR

PROCESSED_SENTINEL_DIR = os.path.join("data", "processed", "sentinel2")
LABEL_DIR = os.path.join("data", "processed", "labels")

def rasterize_hotspots_to_match(firms_csv_path, reference_tif_path, out_mask_path):
    """Mengubah koordinat titik hotspot menjadi piksel biner pada resolusi citra referensi."""
    # 1. Baca data hotspot FIRMS
    df = pd.read_csv(firms_csv_path)
    if df.empty:
        print("  [Info] Data CSV FIRMS kosong, tidak ada titik untuk dirasterisasi.")
        return False
        
    # 2. Buka citra referensi (NBR) untuk mengambil metadata koordinat (Geotransform & CRS)
    with rasterio.open(reference_tif_path) as ref:
        meta = ref.meta.copy()
        transform = ref.transform
        shape = ref.shape  # (height, width)
        
        # Ambil rentang waktu akuisisi citra Sentinel-2 dari nama filenya
        # Format Sentinel-2: ..._YYYYMMDDThhmmss_...
        # Kita akan menyaring hotspot yang aktif di sekitar tanggal citra tersebut
        scene_name = os.path.basename(reference_tif_path)
        try:
            # Mengambil substring tanggal (misal: '20190703')
            date_part = scene_name.split("_")[2][:8]
            scene_date = pd.to_datetime(date_part, format="%Y%m%d")
            print(f"  Tanggal citra referensi terdeteksi: {scene_date.strftime('%Y-%m-%d')}")
        except Exception:
            scene_date = None

        # 3. Filter data hotspot agar hanya mengambil yang sinkron dengan waktu citra (toleransi +/- 3 hari)
        if scene_date and "acq_date" in df.columns:
            df["acq_date"] = pd.to_datetime(df["acq_date"])
            margin = pd.Timedelta(days=3)
            df_filtered = df[(df["acq_date"] >= scene_date - margin) & (df["acq_date"] <= scene_date + margin)]
            print(f"  Menyaring hotspot di rentang tanggal citra: ditemukan {len(df_filtered)} titik dari total {len(df)}.")
        else:
            df_filtered = df

        if df_filtered.empty:
            print("  [skip] Tidak ada hotspot yang aktif di sekitar tanggal citra ini.")
            return False

        # 4. Ubah koordinat Lat/Lon dari FIRMS menjadi objek geometri Shapely Point
        # PENTING: FIRMS menggunakan CRS EPSG:4326 (WGS84)
        points = [Point(lon, lat) for lat, lon in zip(df_filtered["latitude"], df_filtered["longitude"])]
        
        # Jika citra Sentinel Anda menggunakan proyeksi UTM (misal EPSG:32647 atau 32648), 
        # rasterio.features.rasterize secara otomatis menyelaraskannya selama transformasinya tepat.
        # Kita buat pasangan (geometri, nilai_piksel). Nilai 1 berarti area kebakaran/hotspot.
        shapes = [(point, 1) for point in points]
        
        # 5. Lakukan proses pembakaran koordinat menjadi matriks array (Rasterisasi)
        # Piksel default diisi 0 (aman), jika ada titik hotspot diisi 1 (kebakaran)
        mask_array = rasterize(
            shapes=shapes,
            out_shape=shape,
            transform=transform,
            fill=0,
            all_touched=True, # Menyalakan piksel terdekat agar tidak terlalu kecil (resolusi 10m)
            dtype=np.uint8
        )
        
        # 6. Perbarui metadata berkas output masker (tipe data biner cukup uint8 agar ringan)
        meta.update({
            "driver": "GTiff",
            "dtype": "uint8",
            "count": 1,
            "nodata": 0
        })
        
        # Simpan matriks biner label ke folder processed
        with rasterio.open(out_mask_path, "w", **meta) as dst:
            dst.write(mask_array, 1)
            
    print(f"  [sukses] Masker label berhasil dibuat: {out_mask_path}")
    return True

def main():
    os.makedirs(LABEL_DIR, exist_ok=True)
    
    # Cari semua file citra NBR hasil preprocessing sebelumnya
    nbr_files = glob.glob(os.path.join(PROCESSED_SENTINEL_DIR, "*_NBR.tif"))
    if not nbr_files:
        print(f"Tidak ada citra NBR ditemukan di {PROCESSED_SENTINEL_DIR}. Jalankan preprocess_sentinel2.py dulu.")
        return

    print(f"Menemukan {len(nbr_files)} citra referensi NBR untuk dibuatkan maskernya.\n")
    
    for nbr_path in nbr_files:
        scene_name = Path(nbr_path).name
        print(f"Memproses masker untuk: {scene_name}")
        
        # Ekstrak tahun dari nama file citra untuk dicocokkan dengan file CSV FIRMS tahun tersebut
        # Contoh nama file: S2A_MSIL2A_20190703T..._NBR.tif
        try:
            year = scene_name.split("_")[2][:4]
        except Exception:
            print("  [Error] Gagal membaca format tahun dari nama file citra.")
            continue
            
        # Cari file CSV FIRMS yang cocok dengan tahun citra tersebut
        firms_matches = glob.glob(os.path.join(RAW_FIRMS_DIR, f"firms_*_{year}.csv"))
        if not firms_matches:
            print(f"  [warn] File CSV FIRMS untuk tahun {year} tidak ditemukan di {RAW_FIRMS_DIR}. Lewati.")
            continue
            
        out_mask_tiff = os.path.join(LABEL_DIR, scene_name.replace("_NBR.tif", "_MASK.tif"))
        
        # Jalankan fungsi utama rasterisasi
        rasterize_hotspots_to_match(firms_matches[0], nbr_path, out_mask_tiff)
        print("-" * 60)

if __name__ == "__main__":
    main()