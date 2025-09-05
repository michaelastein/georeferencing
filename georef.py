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

# Base camera-to-gimbal rotation matrix
R_C_to_G = np.array([[0, 0, 1],
                    [1, 0, 0],
                    [0, 1, 0]])

T_C_to_G = np.zeros((3, 1))
R_G_to_UAS = np.eye(3)
T_G_to_UAS = np.array([[0.02], [0.0], [0.20]])  # gimbal -> UAS
R_NED_to_ENU = np.array([[1, 0, 0],
                         [0, -1, 0],
                         [0, 0, 1]])

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



def pixel_to_ENU_quat(u, v, width, height, drone_gps, drone_alt, alt_above_ground,
                      yaw, pitch, roll, K=K):
    # Pixel -> normalized camera ray
    i = np.array([[u], [v], [1.0]])
    P_C_dir = np.linalg.inv(K) @ i
    P_C_dir /= np.linalg.norm(P_C_dir)

    # Rotate camera ray into NED frame (using drone attitude)
    q_rot = rotation_quaternion_yaw_pitch_roll(yaw, pitch, roll)
    P_NED_dir = q_rot.rotate(P_C_dir.flatten()).reshape(3, 1)

    # Ray origin = drone position in NED
    pos_ned = np.zeros((3, 1))  # IMU == Camera
    pos_enu = R_NED_to_ENU @ pos_ned
    ray_dir_enu = R_NED_to_ENU @ P_NED_dir

    # Convert drone GPS to UTM
    drone_lat, drone_lon = drone_gps
    zone = int((drone_lon + 180.0) / 6.0) + 1
    epsg_code = 32600 + zone if drone_lat >= 0 else 32700 + zone
    utm_crs = pyproj.CRS.from_epsg(epsg_code)
    transformer_to_utm = pyproj.Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    transformer_from_utm = pyproj.Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
    UTM_x, UTM_y = transformer_to_utm.transform(drone_lon, drone_lat)
    origin_enu = np.array([[UTM_x], [UTM_y], [drone_alt]])
    ray_origin = origin_enu

    # Intersect with ground plane
    ground_z = drone_alt - alt_above_ground
    denom = ray_dir_enu[2, 0]
    if abs(denom) < 1e-9:
        raise ValueError("Ray parallel to ground; cannot intersect.")
    t = (ground_z - ray_origin[2, 0]) / denom
    intersection_enu = ray_origin + t * ray_dir_enu

    # Convert back to lat/lon
    lon_out, lat_out = transformer_from_utm.transform(intersection_enu[0, 0], intersection_enu[1, 0])
    return (lat_out, lon_out), intersection_enu.flatten()



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
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

    u, v = select_pixel(img_array)
    drone_gps = (drone_lat, drone_lon)
    target_gps, enu_pt = pixel_to_ENU_quat(u, v, width, height, drone_gps, drone_alt,
                                           alt_above_ground, yaw, pitch, roll)
    print("Target GPS:", target_gps)

    # Check corner points
    corners = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
    corner_gps = [pixel_to_ENU_quat(x, y, width, height, drone_gps, drone_alt,
                                    alt_above_ground, yaw, pitch, roll)[0] for x, y in corners]

    plotting(
        target_gps=target_gps,
        corner_gps=corner_gps,
        drone_gps=drone_gps
    )

    cv2.circle(img_array, (u, v), radius=5, color=(0, 0, 255), thickness=-1)
    show_image_with_buttons(img_array, u, v)
