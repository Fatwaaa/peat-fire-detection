"""
scripts/download_firms.py

Mengunduh data hotspot titik api (FIRMS - NASA) untuk area Riau,
dipakai sebagai label/ground truth untuk validasi model.

Cara pakai:
    1. Salin .env.example -> .env, isi FIRMS_MAP_KEY
       (daftar gratis di https://firms.modaps.eosdis.nasa.gov/api/map_key/)
    2. python scripts/download_firms.py --year 2023

Catatan:
    - FIRMS Area API hanya menyediakan data historis hingga ~2 tahun ke
      belakang per request kecil; untuk rentang panjang (2019-2023) skrip
      ini melakukan query per-tahun dan menggabungkannya jadi satu CSV.
    - Dokumentasi resmi: https://firms.modaps.eosdis.nasa.gov/api/
"""

import os
import sys
import argparse
import requests
import pandas as pd
from pathlib import Path
from io import StringIO
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import RIAU_BBOX, YEARS, FIRMS_SOURCE, RAW_FIRMS_DIR

load_dotenv()

FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"


def fetch_firms_year(map_key: str, bbox: dict, source: str, year: int) -> pd.DataFrame:
    """
    Mengunduh data hotspot menggunakan Area API NASA FIRMS dengan bounding box
    Riau langsung (bukan Country API - Country API ternyata menolak MAP_KEY ini
    dengan 'Invalid API call' bahkan untuk source _NRT sekalipun, jadi memang
    bukan soal source/tanggal, tapi endpoint country-nya sendiri yang gagal).
    Area API dengan bbox eksplisit sudah terverifikasi bekerja (status 200).
    """
    import time

    area_base_url = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
    # Area API butuh format "west,south,east,north"
    area_coords = f"{bbox['lon_min']},{bbox['lat_min']},{bbox['lon_max']},{bbox['lat_max']}"

    # Tetap gunakan sub-rentang 5 hari untuk kepatuhan batas limit akun gratis
    intervals = [
        # Juli (31 hari)
        (7, 1, 5), (7, 6, 5), (7, 11, 5), (7, 16, 5), (7, 21, 5), (7, 26, 5), (7, 31, 1),
        # Agustus (31 hari)
        (8, 1, 5), (8, 6, 5), (8, 11, 5), (8, 16, 5), (8, 21, 5), (8, 26, 5), (8, 31, 1),
        # September (30 hari)
        (9, 1, 5), (9, 6, 5), (9, 11, 5), (9, 16, 5), (9, 21, 5), (9, 26, 5),
        # Oktober (31 hari)
        (10, 1, 5), (10, 6, 5), (10, 11, 5), (10, 16, 5), (10, 21, 5), (10, 26, 5), (10, 31, 1)
    ]

    frames = []

    for month, day, day_range in intervals:
        date_str = f"{year}-{month:02d}-{day:02d}"

        url = f"{area_base_url}/{map_key}/{source}/{area_coords}/{day_range}/{date_str}"

        time.sleep(1)  # Jeda aman anti-rate limit

        resp = requests.get(url)
        if resp.status_code != 200:
            print(f"    [warn] Gagal mengambil rentang {date_str} (Status: {resp.status_code})")
            print(f"    [debug] URL  : {url}")
            print(f"    [debug] Body : {resp.text[:300]!r}")
            continue

        if resp.text.strip().startswith("<") or "Invalid" in resp.text[:200]:
            print(f"    [warn] Response tidak valid untuk rentang {date_str}")
            print(f"    [debug] URL  : {url}")
            print(f"    [debug] Body : {resp.text[:300]!r}")
            continue

        df_chunk = pd.read_csv(StringIO(resp.text))
        if not df_chunk.empty and "latitude" in df_chunk.columns:
            frames.append(df_chunk)
            print(f"    Berhasil mengambil {len(df_chunk)} titik hotspot untuk rentang {date_str}")
        else:
            print(f"    Rentang {date_str}: 0 titik hotspot")

    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()

def main():
    parser = argparse.ArgumentParser(description="Download data hotspot FIRMS untuk AOI Riau")
    parser.add_argument("--year", type=int, default=None,
                         help="Tahun tertentu saja (default: semua tahun di config.py)")
    args = parser.parse_args()

    map_key = os.getenv("FIRMS_MAP_KEY")
    if not map_key:
        print("ERROR: FIRMS_MAP_KEY belum diisi di file .env")
        print("Daftar gratis di: https://firms.modaps.eosdis.nasa.gov/api/map_key/")
        sys.exit(1)

    os.makedirs(RAW_FIRMS_DIR, exist_ok=True)
    years = [args.year] if args.year else YEARS

    # PENTING: untuk data historis (2019-2023) WAJIB pakai source "_SP" (Standard
    # Processing / arsip historis terkalibrasi). Source "_NRT" (Near Real-Time)
    # hanya menyimpan data ~60 hari terakhir dari hari ini, jadi TIDAK PERNAH
    # akan punya data tahun-tahun lampau, berapa pun day_range yang dipakai.
    # Nama source yang valid di FIRMS API selalu berakhiran _NRT atau _SP -
    # tidak ada source bernama "VIIRS_SNPP" tanpa akhiran (itu pemicu error 400
    # sebelumnya).
    source_validated = FIRMS_SOURCE
    if not source_validated.endswith(("_NRT", "_SP")):
        source_validated = "VIIRS_SNPP_SP"
    elif source_validated.endswith("_NRT"):
        # Otomatis ganti ke versi _SP untuk query data historis (>60 hari lalu)
        source_validated = source_validated.replace("_NRT", "_SP")

    for year in years:
        print(f"Mengunduh hotspot {source_validated} tahun {year}...")
        df = fetch_firms_year(map_key, RIAU_BBOX, source_validated, year)
        
        # Validasi jika dataframe kosong agar file CSV kosong tidak corrupt saat dibaca nanti
        if not df.empty:
            out_path = os.path.join(RAW_FIRMS_DIR, f"firms_{source_validated}_{year}.csv")
            df.to_csv(out_path, index=False)
            print(f"  -> {len(df)} titik hotspot disimpan ke {out_path}")
        else:
            print(f"  -> [Info] Tidak ada data hotspot yang tersimpan untuk tahun {year}.")

    print("\nSelesai.")


if __name__ == "__main__":
    main()