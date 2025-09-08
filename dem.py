import rasterio
from rasterio.transform import rowcol

# Open DEM
dem_path = "output_hh.tif"
with rasterio.open(dem_path) as dem:
    dem_array = dem.read(1)  # Elevation values
    transform = dem.transform

# Coordinates (lon, lat)
x, y = -8.972313899997387, 39.062746999999995

# Convert coordinates to row, col in the DEM array
row, col = rowcol(transform, x, y)

# Extract elevation
elevation = float(dem_array[row, col])
print(f"Elevation at ({x}, {y}): {elevation} m")

