import numpy as np
import pyproj
import cv2
import tkinter as tk
from tkinter import filedialog
import sys
from PIL import Image, ImageTk
import piexif
from pyquaternion import Quaternion
from plot_gps import plotting

# ---------------- Camera Parameters ----------------
K = np.array([[765.0, 0, 320.0],
              [0, 760.0, 256.0],
              [0, 0, 1.0]])

# Base camera-to-gimbal rotation matrix (kept for compatibility)
R_C_to_G = np.array([[0, 0, 1],
                    [1, 0, 0],
                    [0, 1, 0]])

T_C_to_G = np.zeros((3, 1))
R_G_to_UAS = np.eye(3)
T_G_to_UAS = np.array([[0.02], [0.0], [0.20]])  # gimbal -> UAS
R_NED_to_ENU = np.array([[1, 0, 0],
                         [0, -1, 0],
                         [0, 0, 1]])

# Beispiel: Drift aus bekannten Ist/Soll-Paaren
drift_lat = 39.062581769582636 - 39.062750288415195  # Soll - Ist
drift_lon = -8.972569592861582 - (-8.972633100530118)


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
    """
    Generate quaternion from yaw/pitch/roll (aerospace convention).
    - yaw around Z
    - pitch around Y
    - roll around X
    - Order: R = Rz(yaw) * Ry(pitch) * Rx(roll)  (intrinsic ZYX)
    """
    y = np.radians(yaw_deg)
    p = np.radians(pitch_deg)
    r = np.radians(roll_deg)

    q_yaw   = Quaternion(axis=[0, 0, 1], angle=y)  # Z
    q_pitch = Quaternion(axis=[0, 1, 0], angle=p)  # Y
    q_roll  = Quaternion(axis=[1, 0, 0], angle=r)  # X

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


# ----------------- New math functions (replacing incorrect math) -----------------

def pixel_dir_from_K(u, v, K):
    """Compute a unit ray direction in the camera frame from pixel coordinates and intrinsics K.
    Uses the convention that the camera frame has +x right, +y down, +z forward (through the lens).
    """
    pix = np.array([u, v, 1.0])
    dir_cam = np.linalg.inv(K) @ pix
    dir_cam = dir_cam.flatten()
    dir_cam = dir_cam / np.linalg.norm(dir_cam)
    return dir_cam


def rotation_matrix_from_rpy(roll_deg: float, pitch_deg: float, yaw_deg: float):
    """Build rotation matrix from camera -> world (ENU) using the user's conventions.

    Conventions (as provided):
      - 0 yaw = North, 90 yaw = East (clockwise from North)
      - 0 pitch & 0 roll -> camera looks straight down
      - positive pitch: camera front up (tilt toward North)
      - positive roll: camera left up (tilt toward West)

    Returns R such that v_world = R @ v_cam, where world is ENU (X=East, Y=North, Z=Up).
    """
    r = np.radians(roll_deg)
    p = np.radians(pitch_deg)
    y = np.radians(yaw_deg)

    # Roll around camera x (right)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(r), -np.sin(r)],
        [0, np.sin(r), np.cos(r)]
    ])

    # Pitch around camera y (down)
    Ry = np.array([
        [np.cos(p), 0, np.sin(p)],
        [0, 1, 0],
        [-np.sin(p), 0, np.cos(p)]
    ])

    # Yaw is defined clockwise from North (user). Convert to standard CCW by negating angle for matrix.
    cy = np.cos(-y)
    sy = np.sin(-y)
    Rz = np.array([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1]
    ])

    # Base mapping that maps camera frame (x right, y down, z forward) at zero attitude to ENU when
    # camera looks straight down: camera +x -> East, camera +y (down on sensor) -> -North, camera +z -> -Up
    R0 = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])

    R_cam_att = Rx @ Ry
    R = Rz @ R0 @ R_cam_att
    return R


def intersect_ray_with_plane(ray_origin, ray_dir, ground_z):
    """Intersect ray with horizontal plane z = ground_z. Ray: ray_origin + t*ray_dir.
    Returns intersection point (3,) in same units as ray_origin, or raises ValueError if no intersection.
    """
    dz = ray_dir[2]
    if abs(dz) < 1e-9:
        raise ValueError("Ray parallel to ground; cannot intersect.")
    t = (ground_z - ray_origin[2]) / dz
    if t <= 0:
        raise ValueError("Intersection is behind the camera (t <= 0).")
    return ray_origin + t * ray_dir


# ---------------- Main corrected projection routine ----------------

def pixel_to_ENU_quat(u, v, width, height, drone_gps, drone_alt, alt_above_ground,
                      yaw, pitch, roll, K=K):
    """Map pixel (u,v) to GPS (lat, lon) and intersection in UTM/ENU.

    This function replaces the old math with a direct-georeferencing pipeline:
      - get camera ray from intrinsics (K)
      - rotate ray into ENU using the provided yaw/pitch/roll and the camera-to-world mapping
      - intersect with ground plane at height (drone_alt - alt_above_ground)
      - convert UTM<->WGS84 with pyproj
    """
    # 1) Ray in camera frame (unit vector)
    dir_cam = pixel_dir_from_K(u, v, K)  # shape (3,)

    # 2) Rotation camera -> ENU using yaw/pitch/roll (user conventions)
    R = rotation_matrix_from_rpy(roll, pitch, yaw)
    dir_enu = R @ dir_cam  # unit direction in ENU

    # 3) Convert drone lat/lon -> UTM (we'll work in UTM meters for ENU horizontal coords)
    drone_lat, drone_lon = drone_gps
    zone = int((drone_lon + 180.0) / 6.0) + 1
    epsg_code = 32600 + zone if drone_lat >= 0 else 32700 + zone
    utm_crs = pyproj.CRS.from_epsg(epsg_code)
    transformer_to_utm = pyproj.Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    transformer_from_utm = pyproj.Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
    UTM_x, UTM_y = transformer_to_utm.transform(drone_lon, drone_lat)

    # 4) Ray origin in UTM/ENU coordinates (meters)
    ray_origin = np.array([UTM_x, UTM_y, drone_alt], dtype=float)

    # 5) Intersect with ground plane at z = drone_alt - alt_above_ground
    ground_z = drone_alt - alt_above_ground
    try:
        intersection = intersect_ray_with_plane(ray_origin, dir_enu, ground_z)
    except ValueError as e:
        raise

    # 6) Convert back to lat/lon
    utm_east = float(intersection[0])
    utm_north = float(intersection[1])
    lon_out, lat_out = transformer_from_utm.transform(utm_east, utm_north)

    return (lat_out, lon_out), intersection


def show_image_with_buttons(img_array, u, v):
    """
    Display an image with rotation buttons.
    A red dot is drawn at pixel (u, v).
    Closing the window will also terminate the program.
    """

    # Draw marker on the image
    img_with_dot = img_array.copy()
    cv2.circle(img_with_dot, (u, v), radius=5, color=(0, 0, 255), thickness=-1)

    # Convert to PIL image (RGB)
    pil_img = Image.fromarray(cv2.cvtColor(img_with_dot, cv2.COLOR_BGR2RGB))

    # Create Tkinter window
    root = tk.Tk()
    root.title("Image Viewer")

    # Store current image in a dictionary
    state = {"img": pil_img}

    # Label for displaying the image
    canvas = tk.Label(root)
    canvas.pack()

    def update_image():
        """Update the displayed image."""
        tk_img = ImageTk.PhotoImage(state["img"], master=root)
        canvas.configure(image=tk_img)
        canvas.image = tk_img  # Keep reference alive

    def rotate_left():
        """Rotate the image 90° counter-clockwise."""
        state["img"] = state["img"].rotate(90, expand=True)
        update_image()

    def rotate_right():
        """Rotate the image 90° clockwise."""
        state["img"] = state["img"].rotate(-90, expand=True)
        update_image()

    def on_close():
        """Exit the entire program when the window is closed."""
        root.destroy()
        sys.exit(0)

    # Handle window close event
    root.protocol("WM_DELETE_WINDOW", on_close)

    # Button frame
    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=10)

    tk.Button(btn_frame, text="⟲ Rotate Left", command=rotate_left).pack(side=tk.LEFT, padx=5)
    tk.Button(btn_frame, text="⟳ Rotate Right", command=rotate_right).pack(side=tk.LEFT, padx=5)

    # Show first image
    update_image()

    # Run Tkinter event loop
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
    print("Drone GPS:", drone_lat, drone_lon, drone_alt)
    print("Yaw/Pitch/Roll (deg):", yaw, pitch, roll)
    print("Alt above ground (m):", alt_above_ground)

    img_array = np.array(img)
    if img_array.dtype == np.uint16:
        img_array = (img_array / 256).astype(np.uint8)
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

    u, v = select_pixel(img_array)
    drone_gps = (drone_lat, drone_lon)
    target_gps, enu_pt = pixel_to_ENU_quat(u, v, width, height, drone_gps, drone_alt,
                                           alt_above_ground, yaw, pitch, roll)
    print("Target GPS:", target_gps)

    # Check corner points (robust: skip corners that fail)
    corners = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
    corner_gps = []
    for x, y in corners:
        try:
            gps, _ = pixel_to_ENU_quat(x, y, width, height, drone_gps, drone_alt,
                                       alt_above_ground, yaw, pitch, roll)
            corner_gps.append(gps)
        except Exception as e:
            corner_gps.append(None)

    plotting(
        target_gps=target_gps,
        corner_gps=corner_gps,
        drone_gps=drone_gps
    )

    cv2.circle(img_array, (u, v), radius=5, color=(0, 0, 255), thickness=-1)
    show_image_with_buttons(img_array, u, v)
