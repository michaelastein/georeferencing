import numpy as np
import pyproj
import cv2
import tkinter as tk
from tkinter import filedialog
import sys
import os
from PIL import Image, ImageTk
import piexif
from plot_maps import plot_google_maps
from plot_cad import plot_cad_map  # optional

# ---------------- Camera Parameters ----------------
K = np.array([[764.7, 0, 320.0],
              [0, 763.9, 256.0],
              [0, 0, 1.0]])

# --- correction in (Forward, Right, Up) relative to drone heading ---
corr_forward = 0
corr_right   = 0
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

def rotation_matrix_from_rpy(roll_deg, pitch_deg, yaw_deg):
    r = np.radians(roll_deg)
    p = np.radians(pitch_deg)
    y = np.radians(yaw_deg)

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(r), -np.sin(r)],
                   [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)],
                   [0, 1, 0],
                   [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0],
                   [np.sin(y), np.cos(y), 0],
                   [0, 0, 1]])

    R_att = Rz @ Ry @ Rx

    # Camera→ENU mapping, jetzt direkt mit 90° Drehung im Uhrzeigersinn
    R_cam2enu = np.array([
        [0, 1, 0],   # Cam X → East
        [1, 0, 0],   # Cam Y → North
        [0, 0, -1]   # Cam Z → Up
    ])


    return R_cam2enu @ R_att


def pixel_dir_from_K(u, v, K):
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    # 90° Drehung im Uhrzeigersinn
    x = (cy - v) / fy  # invertiert Y
    y = (u - cx) / fx  # X bleibt, wird nach Y
    z = 1.0
    dir_cam = np.array([x, y, z])
    return dir_cam / np.linalg.norm(dir_cam)




def intersect_ray_with_plane(ray_origin, ray_dir, plane_z):
    dz = ray_dir[2]
    if abs(dz) < 1e-9:
        dz = 1e-6
    t = (plane_z - ray_origin[2]) / dz
    if t <= 0:
        t = 1e-3
    return ray_origin + t * ray_dir

def apply_heading_relative_offset(intersection_utm, yaw_deg, forward_m=0.0, right_m=0.0, up_m=0.0):
    yaw_rad = np.radians(yaw_deg)
    fwd_enu = np.array([np.sin(yaw_rad), np.cos(yaw_rad), 0])
    right_enu = np.array([np.cos(yaw_rad), -np.sin(yaw_rad), 0])
    up_enu = np.array([0, 0, 1])
    offset = forward_m * fwd_enu + right_m * right_enu + up_m * up_enu
    return intersection_utm + offset

def pixel_to_ENU(u, v, drone_gps, drone_alt, rel_alt, yaw, pitch, roll,
                 K=K, corr_forward_m=0.0, corr_right_m=0.0, corr_up_m=0.0,
                 panel_height_m=2.0):  
    dir_cam = pixel_dir_from_K(u, v, K)
    R = rotation_matrix_from_rpy(roll, pitch, yaw)
    dir_enu = R @ dir_cam

    drone_lat, drone_lon = drone_gps
    zone = int((drone_lon + 180) / 6) + 1
    epsg_code = 32600 + zone if drone_lat >= 0 else 32700 + zone
    utm_crs = pyproj.CRS.from_epsg(epsg_code)
    t_to_utm = pyproj.Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    t_from_utm = pyproj.Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)

    UTM_x, UTM_y = t_to_utm.transform(drone_lon, drone_lat)
    
    # Reduce relative height by panel_height
    ground_elev = drone_alt - (rel_alt - panel_height_m)

    ray_origin = np.array([UTM_x, UTM_y, drone_alt + corr_up_m], dtype=float)

    intersection_raw = intersect_ray_with_plane(ray_origin, dir_enu, ground_elev)
    intersection_corr = apply_heading_relative_offset(intersection_raw, yaw,
                                                     forward_m=corr_forward_m,
                                                     right_m=corr_right_m,
                                                     up_m=corr_up_m)
    lon_out, lat_out = t_from_utm.transform(intersection_corr[0], intersection_corr[1])
    return (lat_out, lon_out), intersection_corr, intersection_raw


# ---------------- EXIF & Image Functions ----------------
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
    yaw = pitch = roll = rel_alt = None
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
                    elif key_lower in ["relativealt"]:
                        rel_alt = float(value)
                except ValueError:
                    pass
    if yaw is None or pitch is None or roll is None:
        raise ValueError("Missing yaw, pitch, or roll in image description.")
    return yaw, pitch, roll, rel_alt

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
    if isinstance(alt_ref, (bytes, bytearray)):
        alt_ref_val = int(alt_ref[0])
    else:
        alt_ref_val = int(alt_ref)
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
    u = clicked_point.get('u', width//2)
    v = clicked_point.get('v', height//2)
    return u, v

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

def latlon_apply_heading_offset(lat, lon, yaw_deg, forward_m=0.0, right_m=0.0, up_m=0.0):
    zone = int((lon + 180)/6)+1
    epsg_code = 32600 + zone if lat >= 0 else 32700 + zone
    utm_crs = pyproj.CRS.from_epsg(epsg_code)
    t_to_utm = pyproj.Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    t_from_utm = pyproj.Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
    utm_x, utm_y = t_to_utm.transform(lon, lat)
    utm_xyz = np.array([utm_x, utm_y, 0], dtype=float)
    utm_corr = apply_heading_relative_offset(utm_xyz, yaw_deg, forward_m, right_m, up_m)
    lon_c, lat_c = t_from_utm.transform(float(utm_corr[0]), float(utm_corr[1]))
    return lat_c, lon_c, utm_corr

# ---------------- Main ----------------
if __name__ == "__main__":
    img, file_path = load_image()
    width, height = img.size
    try:
        exif_dict = piexif.load(img.info['exif']) if 'exif' in img.info else piexif.load(file_path)
    except Exception:
        exif_dict = piexif.load(file_path)

    yaw, pitch, roll, rel_alt = parse_description_from_exif(exif_dict)
    drone_lat, drone_lon, drone_alt = extract_gps_from_exif(exif_dict)

    print("Drone GPS (original):", drone_lat, drone_lon, drone_alt)
    print("Yaw/Pitch/Roll (deg):", yaw, pitch, roll)
    print("Relative altitude:", rel_alt)

    img_array = np.array(img)
    if img_array.dtype == np.uint16:
        img_array = (img_array / 256).astype(np.uint8)
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

    u, v = select_pixel(img_array)
    drone_gps = (drone_lat, drone_lon)

    target_gps, enu_corr, enu_raw = pixel_to_ENU(
        u, v, drone_gps, drone_alt, rel_alt, yaw, pitch, roll,
        corr_forward_m=corr_forward,
        corr_right_m=corr_right,
        corr_up_m=corr_up
    )
    print("Target GPS:", target_gps)

    # Corners
    corners_px = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
    corner_gps = []
    for x, y in corners_px:
        gps, _, _ = pixel_to_ENU(
            x, y, drone_gps, drone_alt, rel_alt, yaw, pitch, roll,
            corr_forward_m=corr_forward,
            corr_right_m=corr_right,
            corr_up_m=corr_up
        )
        corner_gps.append(gps)

    drone_lat_corr, drone_lon_corr, drone_utm_corr = latlon_apply_heading_offset(
        drone_lat, drone_lon, yaw,
        forward_m=corr_forward, right_m=corr_right, up_m=corr_up
    )
    drone_gps_corrected = (drone_lat_corr, drone_lon_corr)

    plot_google_maps(target_gps=target_gps, corner_gps=corner_gps, drone_gps=drone_gps_corrected)
    plot_cad_map(target_gps=target_gps, corner_gps=corner_gps, drone_gps=drone_gps_corrected)

    cv2.circle(img_array, (u,v), radius=5, color=(0,0,255), thickness=-1)
    show_image_with_buttons(img_array, u,v, filename=file_path)
