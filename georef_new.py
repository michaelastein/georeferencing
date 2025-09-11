import numpy as np
import pyproj
import cv2
import tkinter as tk
from tkinter import filedialog
import sys
import os
from PIL import Image, ImageTk
import piexif
from pyquaternion import Quaternion
from plot_maps import plot_google_maps
from plot_cad import plot_cad_map
import rasterio
from rasterio.transform import rowcol

# ---------------- Camera Parameters ----------------
# compute K from FOV
W, H = 640, 512
fx = (W / 2.0) / np.tan(np.radians(45.0 / 2.0))
fy = (H / 2.0) / np.tan(np.radians(37.0 / 2.0))
cx = (W - 1) / 2.0
cy = (H - 1) / 2.0
K = np.array([[fx, 0.0, cx],
              [0.0, fy, cy],
              [0.0, 0.0, 1.0]], dtype=float)





# --- drift / correction in (Forward, Right, Up) depending on drone heading ---
corr_forward = -13
corr_right   = -13
corr_up      = 0.0

panel_height= 2  # optional: add height of solar panels



# ---------------- Load DEM ----------------
dem_path = "output_hh.tif"
dem = rasterio.open(dem_path)


def dem_height(gps):
    lat, lon = gps
    row, col = rowcol(dem.transform, lon, lat)
    row = np.clip(row, 0, dem.height - 1)
    col = np.clip(col, 0, dem.width - 1)
    height = dem.read(1)[row, col]
    return float(height)
# ---------------- Utility Functions ----------------
def rational_to_float(r):
    try:
        return r[0] / r[1]
    except Exception:
        return float(r)

def gps_to_decimal(coord, ref):
    deg = rational_to_float(coord[0])
    minute = rational_to_float(coord[1])
    sec = rational_to_float(coord[2])
    val = deg + minute / 60.0 + sec / 3600.0
    if isinstance(ref, bytes):
        ref = ref.decode(errors='ignore')
    if ref in ['S', 's', 'W', 'w']:
        val = -val
    return val

def rotation_quaternion_yaw_pitch_roll(yaw_deg, pitch_deg, roll_deg):
    y = np.radians(yaw_deg)
    p = np.radians(pitch_deg)
    r = np.radians(roll_deg)
    q_yaw = Quaternion(axis=[0, 0, 1], angle=y)
    q_pitch = Quaternion(axis=[0, 1, 0], angle=p)
    q_roll = Quaternion(axis=[1, 0, 0], angle=r)
    return q_yaw * q_pitch * q_roll

def load_image(file_dialog=True, path=None):
    if file_dialog:
        root = tk.Tk()
        root.withdraw()
        path = tk.filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp")]
        )
        if not path:
            print("No image selected. Exiting.")
            sys.exit(0)
    img = Image.open(path)
    try:
        exif_dict = piexif.load(img.info['exif']) if 'exif' in img.info else piexif.load(path)
    except Exception:
        exif_dict = piexif.load(path)
    return img, path, exif_dict

def parse_description_from_exif(exif_dict):
    desc = exif_dict.get('0th', {}).get(piexif.ImageIFD.ImageDescription, b'')
    if isinstance(desc, bytes):
        desc = desc.decode(errors='ignore')
    yaw = pitch = roll = None
    if desc:
        for part in str(desc).split(","):
            kv = part.strip().split("=")
            if len(kv) == 2:
                key, value = kv
                key_lower = key.strip().lower()
                try:
                    if key_lower == "yaw":
                        yaw = float(value)
                    elif key_lower == "pitch":
                        pitch = float(value)
                    elif key_lower == "roll":
                        roll = float(value)
                except ValueError:
                    pass
    if yaw is None or pitch is None or roll is None:
        raise ValueError("Missing yaw/pitch/roll in EXIF description.")
    return yaw, pitch, roll


def extract_gps_from_exif(exif_dict):
    gps_ifd = exif_dict.get("GPS", {})
    if not gps_ifd:
        raise ValueError("Missing GPS IFD in EXIF.")
    lat_tag = gps_ifd.get(piexif.GPSIFD.GPSLatitude)
    lat_ref = gps_ifd.get(piexif.GPSIFD.GPSLatitudeRef)
    lon_tag = gps_ifd.get(piexif.GPSIFD.GPSLongitude)
    lon_ref = gps_ifd.get(piexif.GPSIFD.GPSLongitudeRef)
    alt_tag = gps_ifd.get(piexif.GPSIFD.GPSAltitude)
    alt_ref = gps_ifd.get(piexif.GPSIFD.GPSAltitudeRef, 0)
    if not (lat_tag and lat_ref and lon_tag and lon_ref and alt_tag is not None):
        raise ValueError("Missing GPS fields (lat/lon/alt) in EXIF.")
    lat = gps_to_decimal(lat_tag, lat_ref)
    lon = gps_to_decimal(lon_tag, lon_ref)
    alt = rational_to_float(alt_tag)
    try:
        if isinstance(alt_ref, (bytes, bytearray)):
            alt_ref_val = int(alt_ref[0])
        else:
            alt_ref_val = int(alt_ref)
    except Exception:
        alt_ref_val = 0
    if alt_ref_val == 1:
        alt = -alt
    return lat, lon, alt

def select_pixel(img_array):
    clicked_point = {}
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_point['u'] = x
            clicked_point['v'] = y
            cv2.destroyAllWindows()
    cv2.imshow("Click on target pixel (press ESC to skip)", img_array)
    cv2.setMouseCallback("Click on target pixel (press ESC to skip)", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    width, height = img_array.shape[1], img_array.shape[0]
    u = clicked_point.get('u', width // 2)
    v = clicked_point.get('v', height // 2)
    return u, v



# ---------------- Math functions ----------------
def pixel_dir_from_K(u, v, K):
    pix = np.array([u, v, 1.0])
    dir_cam = np.linalg.inv(K) @ pix
    dir_cam = dir_cam.flatten()
    return dir_cam / np.linalg.norm(dir_cam)

def rotation_matrix_from_rpy(roll_deg: float, pitch_deg: float, yaw_deg: float):
    r = np.radians(roll_deg)
    p = np.radians(pitch_deg)
    y = np.radians(yaw_deg)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(r), -np.sin(r)],
                   [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)],
                   [0, 1, 0],
                   [-np.sin(p), 0, np.cos(p)]])
    cy = np.cos(-y)
    sy = np.sin(-y)
    Rz = np.array([[cy, -sy, 0],
                   [sy, cy, 0],
                   [0, 0, 1]])
    R0 = np.array([[1, 0, 0],
                   [0, -1, 0],
                   [0, 0, -1]])
    R_cam_att = Rx @ Ry
    return Rz @ R0 @ R_cam_att

def intersect_ray_with_plane(ray_origin, ray_dir, ground_z):
    dz = ray_dir[2]
    if abs(dz) < 1e-9:
            raise ValueError("Ray parallel to ground; cannot intersect.")
    t = (ground_z - ray_origin[2]) / dz
    if t <= 0:
        raise ValueError("Intersection is behind the camera (t <= 0).")
    return ray_origin + t * ray_dir

# --- General correction relative to drone heading ---
def apply_heading_relative_offset(intersection_utm, yaw_deg,
                                  forward_m=0.0, right_m=0.0, up_m=0.0):
    yaw_rad = np.radians(yaw_deg)
    fwd_enu = np.array([np.sin(yaw_rad), np.cos(yaw_rad), 0.0])
    right_enu = np.array([np.cos(yaw_rad), -np.sin(yaw_rad), 0.0])
    up_enu = np.array([0.0, 0.0, 1.0])
    offset = forward_m * fwd_enu + right_m * right_enu + up_m * up_enu
    return intersection_utm + offset

def latlon_apply_heading_offset(lat, lon, yaw_deg, forward_m=0.0, right_m=0.0, up_m=0.0):
    zone = int((lon + 180.0) / 6.0) + 1
    epsg_code = 32600 + zone if lat >= 0 else 32700 + zone
    utm_crs = pyproj.CRS.from_epsg(epsg_code)
    t_to_utm = pyproj.Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    t_from_utm = pyproj.Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
    utm_x, utm_y = t_to_utm.transform(lon, lat)
    utm_z = 0.0
    utm_xyz = np.array([utm_x, utm_y, utm_z], dtype=float)
    utm_corrected = apply_heading_relative_offset(utm_xyz, yaw_deg, forward_m=forward_m, right_m=right_m, up_m=up_m)
    lon_c, lat_c = t_from_utm.transform(float(utm_corrected[0]), float(utm_corrected[1]))
    return lat_c, lon_c

def get_corrected_drone_gps(exif_dict, forward_m=corr_forward, right_m=corr_right, up_m=corr_up):
    drone_lat, drone_lon, drone_alt = extract_gps_from_exif(exif_dict)
    yaw, pitch, roll = parse_description_from_exif(exif_dict)
    drone_gps_corrected = latlon_apply_heading_offset(drone_lat, drone_lon, yaw,
                                                      forward_m=forward_m,
                                                      right_m=right_m,
                                                      up_m=up_m)
    return drone_gps_corrected, (drone_lat, drone_lon, drone_alt), (yaw, pitch, roll)

def pixel_to_ENU_quat(u, v, drone_gps, drone_alt, yaw, pitch, roll, K=K,
    corr_forward_m=0.0, corr_right_m=0.0, corr_up_m=0.0,
    panel_height_m=panel_height):  # optional: add height of solar panels
    dir_cam = pixel_dir_from_K(u, v, K)
    R = rotation_matrix_from_rpy(roll, pitch, yaw)
    dir_enu = R @ dir_cam

    drone_lat, drone_lon = drone_gps
    zone = int((drone_lon + 180.0) / 6.0) + 1
    epsg_code = 32600 + zone if drone_lat >= 0 else 32700 + zone
    utm_crs = pyproj.CRS.from_epsg(epsg_code)
    t_to_utm = pyproj.Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    t_from_utm = pyproj.Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)

    # Convert drone GPS to UTM
    UTM_x, UTM_y = t_to_utm.transform(drone_lon, drone_lat)

    # Use DEM + 1m extra for solar panels
    ground_elev = dem_height(drone_lat, drone_lon) + panel_height_m

    ray_origin = np.array([UTM_x, UTM_y, drone_alt], dtype=float)
    ground_z = ground_elev

    intersection_raw = intersect_ray_with_plane(ray_origin, dir_enu, ground_z)

    intersection_corr = apply_heading_relative_offset(
        intersection_raw, yaw,
        forward_m=corr_forward_m,
        right_m=corr_right_m,
        up_m=corr_up_m
    )

    lon_out, lat_out = t_from_utm.transform(intersection_corr[0], intersection_corr[1])
    return (lat_out, lon_out), intersection_corr, intersection_raw


def show_image_with_buttons(img_array, u, v, filename):
    img_with_dot = img_array.copy()
    cv2.circle(img_with_dot, (u, v), radius=5, color=(0, 0, 255), thickness=-1)
    pil_img = Image.fromarray(cv2.cvtColor(img_with_dot, cv2.COLOR_BGR2RGB))
    root = tk.Tk()
    root.title(os.path.basename(filename))
    state = {"img": pil_img}
    canvas = tk.Label(root)
    canvas.pack()
    def update_image():
        tk_img = ImageTk.PhotoImage(state["img"], master=root)
        canvas.configure(image=tk_img)
        canvas.image = tk_img
    def rotate_left():
        state["img"] = state["img"].rotate(90, expand=True)
        update_image()
    def rotate_right():
        state["img"] = state["img"].rotate(-90, expand=True)
        update_image()
    def on_close():
        root.destroy()
        sys.exit(0)
    root.protocol("WM_DELETE_WINDOW", on_close)
    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=10)
    tk.Button(btn_frame, text="⟲ Rotate Left", command=rotate_left).pack(side=tk.LEFT, padx=5)
    tk.Button(btn_frame, text="⟳ Rotate Right", command=rotate_right).pack(side=tk.LEFT, padx=5)
    update_image()
    root.mainloop()

if __name__ == "__main__":
    # ---------------- Load image and EXIF ----------------
    img, file_path, exif_dict = load_image()
    img_array = np.array(img)
    if img_array.dtype == np.uint16:
        img_array = (img_array / 256).astype(np.uint8)
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

    # ---------------- Compute corrected drone GPS ----------------
    drone_gps_corrected, drone_gps_raw, drone_pose = get_corrected_drone_gps(exif_dict)
    yaw, pitch, roll = drone_pose
    print("Drone GPS (original):", drone_gps_raw)
    print("Corrected Drone GPS:", drone_gps_corrected)
    print("Drone yaw/pitch/roll:", drone_pose)

    # ---------------- DEM height under drone ----------------
    ground_elev = dem_height(drone_gps_corrected)
    print("DEM height under drone (ellipsoid):", ground_elev, "m")

    # ---------------- Interactive pixel selection ----------------
    u, v = select_pixel(img_array)
    print("Selected pixel:", u, v)

    # ---------------- Compute target GPS from selected pixel ----------------
    target_gps, enu_corr, enu_raw = pixel_to_ENU_quat(
        u, v,
        drone_gps_raw[:2],
        drone_gps_raw[2],
        yaw, pitch, roll,
        corr_forward_m=corr_forward,
        corr_right_m=corr_right,
        corr_up_m=corr_up,
        panel_height_m=panel_height
    )
    print("Target GPS (corrected with DEM):", target_gps)

    # ---------------- Compute corner GPS ----------------
    width, height = img.size
    corners_px = [(0, 0), (width-1, 0), (width-1, height-1), (0, height-1)]
    corner_gps = []
    for x, y in corners_px:
        try:
            gps, _, _ = pixel_to_ENU_quat(
                x, y,
                drone_gps_raw[:2],
                drone_gps_raw[2],
                yaw, pitch, roll,
                corr_forward_m=corr_forward,
                corr_right_m=corr_right,
                corr_up_m=corr_up,
                panel_height_m=panel_height
            )
            corner_gps.append(gps)
        except Exception:
            corner_gps.append(None)

    # ---------------- Plot maps ----------------
    plot_google_maps(target_gps=target_gps, corner_gps=corner_gps, drone_gps=drone_gps_corrected)
    plot_cad_map(target_gps=target_gps, corner_gps=corner_gps, drone_gps=drone_gps_corrected)

    # ---------------- Show image with marked target ----------------
    cv2.circle(img_array, (u, v), radius=5, color=(0, 0, 255), thickness=-1)
    show_image_with_buttons(img_array, u, v, file_path)