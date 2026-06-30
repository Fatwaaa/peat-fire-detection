"""
scripts/download_sentinel2.py

Mencari dan mengunduh citra Sentinel-2 Level-2A untuk area Riau pada
musim kemarau (Jul-Okt) tahun 2019-2023, mengikuti spesifikasi proposal.

Cara pakai:
    1. Salin .env.example -> .env, isi CDSE_USERNAME dan CDSE_PASSWORD
       (daftar gratis di https://dataspace.copernicus.eu/)
    2. pip install -r requirements.txt
    3. python scripts/download_sentinel2.py --max-scenes 5
       (mulai dengan jumlah kecil dulu untuk uji coba)

Catatan:
    - Satu scene Sentinel-2 L2A penuh berukuran ~800MB-1GB.
    - Skrip ini hanya MENCARI dan MENGUNDUH produk lengkap (.zip / SAFE).
    - Ekstraksi band tertentu dilakukan di tahap preprocessing (skrip lain).
"""

import os
import sys
import argparse
import requests
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import AOI_WKT, YEARS, SEASON_START_MONTH_DAY, SEASON_END_MONTH_DAY, \
    SENTINEL2_PRODUCT_TYPE, MAX_CLOUD_COVER, RAW_SENTINEL2_DIR

load_dotenv()

CATALOGUE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
# Endpoint OAuth yang BENAR untuk Copernicus Data Space Ecosystem (CDSE).
# (Endpoint lama services.sentinel-hub.com/oauth/token TIDAK berlaku untuk
# login username/password CDSE -- itu untuk OAuth client app Sentinel Hub
# yang terpisah dan butuh client_id/client_secret, bukan email/password akun.)
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"


def get_access_token(username: str, password: str) -> str:
    """Ambil access token CDSE menggunakan OAuth Password Grant (email & password akun)."""
    data = {
        "grant_type": "password",
        "username": username,
        "password": password,
        "client_id": "cdse-public",
    }

    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    response = requests.post(TOKEN_URL, data=data, headers=headers)
    if response.status_code != 200:
        print("Gagal mengambil token. Respons server:")
        print(response.text)
    response.raise_for_status()
    return response.json()["access_token"]


def search_products(year: int, max_cloud_cover: int = MAX_CLOUD_COVER, top: int = 20):
    """Cari produk Sentinel-2 L2A untuk satu tahun musim kemarau di AOI Riau."""
    start_date = f"{year}-{SEASON_START_MONTH_DAY}T00:00:00.000Z"
    end_date = f"{year}-{SEASON_END_MONTH_DAY}T23:59:59.999Z"

    # Perbaikan sintaksis query sesuai standar CDSE OData API Terbaru
    filter_query = (
        f"Collection/Name eq 'SENTINEL-2' and "
        f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/Value eq '{SENTINEL2_PRODUCT_TYPE}') and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{AOI_WKT}') and "
        f"ContentDate/Start ge {start_date} and "
        f"ContentDate/Start le {end_date} and "
        f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/Value le {max_cloud_cover})"
    )

    params = {
        "$filter": filter_query,
        "$orderby": "ContentDate/Start asc",
        "$top": top,
        "$format": "json",
    }

    response = requests.get(CATALOGUE_URL, params=params)
    response.raise_for_status()
    return response.json().get("value", [])


def download_product(product_id: str, product_name: str, token: str, out_dir: str):
    """Unduh satu produk berdasarkan ID, simpan sebagai .zip."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{product_name}.zip")

    if os.path.exists(out_path):
        print(f"  [skip] sudah ada: {out_path}")
        return out_path

    url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
    headers = {"Authorization": f"Bearer {token}"}

    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  mengunduh {product_name}: {pct:.1f}%", end="")
        print()

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Download Sentinel-2 L2A untuk AOI Riau")
    parser.add_argument("--max-scenes", type=int, default=5,
                         help="Maksimum jumlah scene yang diunduh (default 5, untuk uji coba)")
    parser.add_argument("--search-only", action="store_true",
                         help="Hanya cari & tampilkan daftar produk, tanpa mengunduh")
    args = parser.parse_args()

    username = os.getenv("CDSE_USERNAME")
    password = os.getenv("CDSE_PASSWORD")
    if not username or not password:
        print("ERROR: CDSE_USERNAME / CDSE_PASSWORD belum diisi di file .env")
        print("Daftar gratis di: https://dataspace.copernicus.eu/")
        sys.exit(1)

    all_products = []
    for year in YEARS:
        print(f"Mencari scene tahun {year} (musim kemarau)...")
        products = search_products(year)
        print(f"  ditemukan {len(products)} scene dengan cloud cover <= {MAX_CLOUD_COVER}%")
        all_products.extend(products)

    print(f"\nTotal scene ditemukan: {len(all_products)}")

    if args.search_only:
        for p in all_products[: args.max_scenes]:
            print(f"  - {p['Name']}  (ID: {p['Id']})")
        return

    if not all_products:
        print("Tidak ada produk ditemukan. Coba naikkan MAX_CLOUD_COVER di config.py.")
        return

    print("Mengambil access token...")
    token = get_access_token(username, password)

    to_download = all_products[: args.max_scenes]
    print(f"Mengunduh {len(to_download)} scene ke {RAW_SENTINEL2_DIR}/ ...")
    for p in to_download:
        download_product(p["Id"], p["Name"], token, RAW_SENTINEL2_DIR)

    print("\nSelesai. Catatan: token CDSE berlaku ~10 menit, skrip akan otomatis")
    print("gagal jika unduhan terlalu lama -- jalankan ulang untuk melanjutkan scene yang belum selesai.")


if __name__ == "__main__":
    main()