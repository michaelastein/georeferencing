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

# ---------------- Camera Parameters ----------------
K = np.array([[765.0, 0, 320.0],
              [0, 760.0, 256.0],
              [0, 0, 1.0]])

R_C_to_G = np.array([[0, 0, 1],
                     [1, 0, 0],
                     [0, 1, 0]])
T_C_to_G = np.zeros((3, 1))
R_G_to_UAS = np.eye(3)
T_G_to_UAS = np.array([[0.02], [0.0], [0.20]])
R_NED_to_ENU = np.array([[1, 0, 0],
                         [0, -1, 0],
                         [0, 0, 1]])

# --- drift / correction in (Forward, Right, Up) depending on drone heading ---
corr_forward = -13.0
corr_right   = -17.5  # example: 20 m left relative to heading
corr_up      = 0.0

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

def load_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp"), ("All files", "*.*")]
    )
    if not file_path:
        print("No image selected. Exiting program.")
        sys.exit(0)
    img = Image.open(file_path)
    print("Loaded:", file_path)
    return img, file_path

def parse_description_from_exif(exif_dict):
    desc = exif_dict.get('0th', {}).get(piexif.ImageIFD.ImageDescription, b'')
    if isinstance(desc, bytes):
        desc = desc.decode(errors='ignore')
    yaw = pitch = roll = alt_above_ground = None
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
                    elif key_lower in ["relativealt", "talt", "alt_above_ground"]:
                        alt_above_ground = float(value)
                except ValueError:
                    pass
    if yaw is None or pitch is None or roll is None or alt_above_ground is None:
        raise ValueError("Missing yaw, pitch, roll, or alt_above_ground in image description.")
    return yaw, pitch, roll, alt_above_ground

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
    return lat_c, lon_c, utm_corrected

def pixel_to_ENU_quat(u, v, drone_gps, drone_alt, alt_above_ground,
                      yaw, pitch, roll, K=K,
                      corr_forward_m=0.0, corr_right_m=0.0, corr_up_m=0.0):
    dir_cam = pixel_dir_from_K(u, v, K)
    R = rotation_matrix_from_rpy(roll, pitch, yaw)
    dir_enu = R @ dir_cam
    drone_lat, drone_lon = drone_gps
    zone = int((drone_lon + 180.0) / 6.0) + 1
    epsg_code = 32600 + zone if drone_lat >= 0 else 32700 + zone
    utm_crs = pyproj.CRS.from_epsg(epsg_code)
    t_to_utm = pyproj.Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    t_from_utm = pyproj.Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
    UTM_x, UTM_y = t_to_utm.transform(drone_lon, drone_lat)
    ray_origin = np.array([UTM_x, UTM_y, drone_alt], dtype=float)
    ground_z = drone_alt - alt_above_ground
    intersection_raw = intersect_ray_with_plane(ray_origin, dir_enu, ground_z)
    intersection_corr = apply_heading_relative_offset(intersection_raw, yaw,
                                                      forward_m=corr_forward_m,
                                                      right_m=corr_right_m,
                                                      up_m=corr_up_m)
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

# ---------------- Main ----------------
if __name__ == "__main__":
    img, file_path = load_image()
    width, height = img.size
    try:
        exif_dict = piexif.load(img.info['exif']) if 'exif' in img.info else piexif.load(file_path)
    except Exception:
        exif_dict = piexif.load(file_path)

    yaw, pitch, roll, alt_above_ground = parse_description_from_exif(exif_dict)
    drone_lat, drone_lon, drone_alt = extract_gps_from_exif(exif_dict)
    print("Drone GPS (original):", drone_lat, drone_lon, drone_alt)
    print("Yaw/Pitch/Roll (deg):", yaw, pitch, roll)
    print("Alt above ground (m):", alt_above_ground)

    img_array = np.array(img)
    if img_array.dtype == np.uint16:
        img_array = (img_array / 256).astype(np.uint8)
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

    u, v = select_pixel(img_array)
    drone_gps = (drone_lat, drone_lon)

    # Compute projections with the chosen correction
    target_gps, enu_corr, enu_raw = pixel_to_ENU_quat(
        u, v, drone_gps, drone_alt,
        alt_above_ground, yaw, pitch, roll,
        corr_forward_m=corr_forward,
        corr_right_m=corr_right,
        corr_up_m=corr_up
    )
    print("Target GPS (corrected):", target_gps)


    # Corners (each corner projection uses same correction)
    corners_px = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
    corner_gps = []
    for x, y in corners_px:
        try:
            gps, enu_c, enu_r = pixel_to_ENU_quat(
                x, y, drone_gps, drone_alt,
                alt_above_ground, yaw, pitch, roll,
                corr_forward_m=corr_forward,
                corr_right_m=corr_right,
                corr_up_m=corr_up
            )
            corner_gps.append(gps)
        except Exception:
            corner_gps.append(None)

    # Shift the drone GPS marker itself for plotting consistency
    drone_lat_corr, drone_lon_corr, drone_utm_corr = latlon_apply_heading_offset(
        drone_lat, drone_lon, yaw,
        forward_m=corr_forward, right_m=corr_right, up_m=corr_up
    )
    drone_gps_corrected = (drone_lat_corr, drone_lon_corr)
    print("Drone GPS (corrected for plotting):", drone_gps_corrected)

    # Plot Google Maps
    plot_google_maps(
        target_gps=target_gps,
        corner_gps=corner_gps,
        drone_gps=drone_gps_corrected
    )
    # To plot CAD map (also uses corrected drone & target positions)
    plot_cad_map(
        target_gps=target_gps,
        corner_gps=corner_gps,
        drone_gps=drone_gps_corrected
    )
    # Draw marker in image and show
    cv2.circle(img_array, (u, v), radius=5, color=(0, 0, 255), thickness=-1)
    show_image_with_buttons(img_array, u, v, filename=file_path)


