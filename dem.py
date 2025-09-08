import rasterio

# Path to your DEM
dem_path = "output_SRTMGL1Ellip.tif"

# GPS points (lat, lon)
gps_points = [
    (39.062746999999995,  -8.972313899997387),
    # add more points as needed
]

# Open the DEM
with rasterio.open(dem_path) as dem:
    dem_band = dem.read(1)  # first band
    for lat, lon in gps_points:
        # Convert GPS (lon, lat) to DEM row/col
        row, col = rasterio.transform.rowcol(dem.transform, lon, lat)
        # Get DEM height at this pixel
        height = dem_band[row, col]
        print(f"Point ({lat}, {lon}) → DEM height: {height:.2f} m")
