"""
config.py
Konfigurasi global untuk project Deteksi Titik Api Gambut.
Ubah nilai di sini sesuai kebutuhan, jangan ubah file script lain.
"""

# --- Area of Interest (AOI) ---
# Bounding box Provinsi Riau (lon_min, lat_min, lon_max, lat_max)
# Sumber kasar: cakupan administratif Riau, Indonesia
RIAU_BBOX = {
    "lon_min": 100.0,
    "lat_min": -1.5,
    "lon_max": 103.5,
    "lat_max": 2.5,
}
# Format: [xmin, ymin, xmax, ymax]
# RIAU_BBOX = [100.0, -1.5, 103.5, 2.5]

# Polygon WKT sederhana dari bbox di atas (dipakai untuk query Copernicus)
def bbox_to_wkt(bbox: dict) -> str:
    return (
        f"POLYGON(({bbox['lon_min']} {bbox['lat_min']}, "
        f"{bbox['lon_max']} {bbox['lat_min']}, "
        f"{bbox['lon_max']} {bbox['lat_max']}, "
        f"{bbox['lon_min']} {bbox['lat_max']}, "
        f"{bbox['lon_min']} {bbox['lat_min']}))"
    )

AOI_WKT = bbox_to_wkt(RIAU_BBOX)

# --- Periode waktu (musim kemarau, sesuai proposal) ---
YEARS = [2019, 2020, 2021, 2022, 2023]
SEASON_START_MONTH_DAY = "07-01"  # Juli
SEASON_END_MONTH_DAY = "10-31"    # Oktober

# --- Parameter pencarian Sentinel-2 ---
SENTINEL2_PRODUCT_TYPE = "S2MSI2A"   # Level-2A (Surface Reflectance)
MAX_CLOUD_COVER = 100                 # persen, sesuaikan jika scene terlalu sedikit

# --- Band yang dipakai (sesuai proposal: RGB + SWIR) ---
BANDS = ["B04", "B08", "B8A", "B12"]  # Red, NIR, NIR-narrow, SWIR2

# --- FIRMS (NASA) ---
# Daftar gratis MAP_KEY di: https://firms.modaps.eosdis.nasa.gov/api/map_key/
FIRMS_SOURCE = "VIIRS_SNPP_SP"   # bisa juga MODIS_NRT, VIIRS_NOAA20_NRT, dst.

# --- Path lokal ---
DATA_DIR = "data"
RAW_SENTINEL2_DIR = f"{DATA_DIR}/raw/sentinel2"
RAW_FIRMS_DIR = f"{DATA_DIR}/raw/firms"
RAW_PEAT_DIR = f"{DATA_DIR}/raw/peat_boundary"
PROCESSED_DIR = f"{DATA_DIR}/processed"