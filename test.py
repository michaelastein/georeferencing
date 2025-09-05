import folium
import webbrowser
import os

# ---------------- Configuration ----------------

# Drone GPS
drone_gps = (39.06280140001979, -8.972497100008285)

# Rotation-order results
rotation_points = {
    "Y*P*R": (39.062867949257615, -8.972597185751093),
    "Y*R*P": (39.06286720160945, -8.972593786369627),
    "P*Y*R": (39.062863203812626, -8.972595370504294),
    "P*R*Y": (39.062733481282166, -8.972632721362944),
    "R*Y*P": (39.06273778938437, -8.972629705792235),
    "R*P*Y": (39.062733484490664, -8.972628962453397),
}

# Currently selected target GPS
target_gps = rotation_points["R*P*Y"]

# Optional: image corners (if available)
corner_gps = None
# Example format:
# corner_gps = [
#     (39.062870, -8.972640),
#     (39.062870, -8.972580),
#     (39.062730, -8.972580),
#     (39.062730, -8.972640)
# ]

# Output map file
map_file = "rotation_points_map.html"

# ---------------- Function ----------------
def plot_points(target_gps, rotation_points=None, corner_gps=None, drone_gps=None, map_file="target_map.html"):
    lat, lon = target_gps

    # Base map centered on target
    m = folium.Map(location=[lat, lon], zoom_start=21, tiles=None)
    google_sat = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}'
    folium.TileLayer(tiles=google_sat, attr='Google', name='Google Satellite',
                     overlay=False, control=False, max_zoom=23).add_to(m)

    # Marker for current target pixel
    folium.Marker([lat, lon], popup=f"Target\nLat: {lat:.7f}\nLon: {lon:.7f}",
                  icon=folium.Icon(color='red', icon='crosshairs')).add_to(m)

    # Mark all rotation order points
    if rotation_points:
        for name, coords in rotation_points.items():
            color = 'green' if coords != target_gps else 'red'
            folium.Marker([coords[0], coords[1]], popup=name,
                          icon=folium.Icon(color=color, icon='info-sign')).add_to(m)
            # Draw line from drone to each rotation point
            if drone_gps:
                folium.PolyLine([drone_gps, coords], color="orange", weight=1.5, opacity=0.7).add_to(m)

    # Polygon connecting image corners if provided
    if corner_gps:
        folium.Polygon(corner_gps + [corner_gps[0]], color="#00FF00", weight=2, fill=False,
                       tooltip="Image Corners").add_to(m)

    # Marker for drone GPS
    if drone_gps:
        d_lat, d_lon = drone_gps
        folium.Marker([d_lat, d_lon], popup=f"Drone\nLat: {d_lat:.7f}\nLon: {d_lon:.7f}",
                      icon=folium.Icon(color='blue', icon='plane')).add_to(m)

    # Save and open map
    m.save(map_file)
    webbrowser.open('file://' + os.path.abspath(map_file))


# ---------------- Main ----------------
if __name__ == "__main__":
    plot_points(target_gps, rotation_points=rotation_points, corner_gps=corner_gps,
                drone_gps=drone_gps, map_file=map_file)
