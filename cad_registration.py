import json
import numpy as np
import cv2
from shapely.geometry import shape
from scipy.optimize import linear_sum_assignment
import georef_new
from PIL import Image
import piexif
import argparse

# ---------------- Parameters ----------------
cad_map = "section_1_ir_cad (1).geojson"

# ---------------- Load CAD polygons ----------------
def load_cad_polygons(geojson_path):
    with open(geojson_path,'r') as f:
        cad_data = json.load(f)
    cad_polygons, cad_ids = [], []
    for i, feat in enumerate(cad_data['features']):
        geom = shape(feat['geometry'])
        if not geom.is_empty:
            coords = np.array(geom.exterior.coords)
            cad_polygons.append(coords)
            cad_ids.append(feat.get("id",i))
    return cad_polygons, cad_ids

# ---------------- IR segmentation ----------------
def segment_solar_panels_ir(image, blur=5, thresh=150, min_area=500):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape)==3 else image
    blur_img = cv2.GaussianBlur(gray,(blur,blur),0)
    _, bin_img = cv2.threshold(blur_img, thresh, 255, cv2.THRESH_BINARY)
    contours,_ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c[:,0,:] for c in contours if cv2.contourArea(c)>=min_area]

# ---------------- Projection ----------------
def project_points_world_to_image(points_xyz, K, R, t):
    homog = np.hstack([points_xyz, np.ones((points_xyz.shape[0],1))])
    proj = K @ (R @ homog.T + t)
    uv = (proj[:2,:]/proj[2:3,:]).T
    return uv

def project_polygons(cad_polygons_world, K, R, t, panel_height=2.0):
    projected = []
    for poly in cad_polygons_world:
        if poly.shape[1]==2:
            poly_xyz = np.hstack([poly,np.full((poly.shape[0],1),panel_height)])
        else:
            poly_xyz = poly
        projected.append(project_points_world_to_image(poly_xyz,K,R,t))
    return projected

# ---------------- Matching ----------------
def bbox_from_poly(poly):
    x_min,y_min = poly.min(axis=0)
    x_max,y_max = poly.max(axis=0)
    return (x_min,y_min,x_max,y_max)

def iou_bbox(a,b):
    ix1,iy1 = max(a[0],b[0]), max(a[1],b[1])
    ix2,iy2 = min(a[2],b[2]), min(a[3],b[3])
    if ix2<=ix1 or iy2<=iy1: return 0.0
    inter = (ix2-ix1)*(iy2-iy1)
    area_a = (a[2]-a[0])*(a[3]-a[1])
    area_b = (b[2]-b[0])*(b[3]-b[1])
    return inter/(area_a+area_b-inter)

def match_detections_to_cad(projected_cad,cad_ids,detections,img_shape):
    proj_bboxes = [bbox_from_poly(p) for p in projected_cad]
    det_bboxes = [bbox_from_poly(np.array(d)) for d in detections]
    P,D = len(proj_bboxes), len(det_bboxes)
    cost = np.full((P,D),1e6,dtype=np.float32)
    diag = np.hypot(*img_shape[::-1])
    for i,cad_bb in enumerate(proj_bboxes):
        cx_c,cy_c = (cad_bb[0]+cad_bb[2])/2, (cad_bb[1]+cad_bb[3])/2
        for j,det_bb in enumerate(det_bboxes):
            cx_d,cy_d = (det_bb[0]+det_bb[2])/2, (det_bb[1]+det_bb[3])/2
            dist = np.hypot(cx_c-cx_d,cy_c-cy_d)
            cost[i,j] = 0.7*(dist/diag)+0.3*(1-iou_bbox(cad_bb,det_bb))
    row,col = linear_sum_assignment(cost)
    return [(cad_ids[r],c) for r,c in zip(row,col) if cost[r,c]<0.5]

# ---------------- Main pipeline ----------------
def main( ir_image_path):
    # Load image & EXIF
    img = Image.open(ir_image_path)
    exif_dict = piexif.load(img.info['exif']) if 'exif' in img.info else piexif.load(ir_image_path)
    img_array = np.array(img)
    if img_array.dtype==np.uint16: img_array=(img_array/256).astype(np.uint8)
    if len(img_array.shape)==2: img_array=cv2.cvtColor(img_array,cv2.COLOR_GRAY2BGR)
    
    # Compute corrected drone GPS and pose
    drone_gps_corr, drone_gps_raw, drone_pose = georef_new.get_corrected_drone_gps(exif_dict)
    yaw,pitch,roll = drone_pose
    print("Drone GPS (original):",drone_gps_raw)
    print("Corrected GPS:",drone_gps_corr)
    print("Drone yaw/pitch/roll:",drone_pose)

    # Segment solar panels in IR
    detections = segment_solar_panels_ir(img_array)

    # Load CAD
    cad_polygons, cad_ids = load_cad_polygons(cad_map)

    # Camera rotation & translation
    R_c2w = georef_new.rotation_matrix_from_rpy(roll,pitch,yaw)
    R = R_c2w.T
    C = np.vstack([drone_gps_corr,drone_gps_raw[2]]).reshape(3,1)
    t = -R@C

    # Project CAD
    projected_cad = project_polygons(cad_polygons, georef_new.K, R, t)

    # Match
    matches = match_detections_to_cad(projected_cad,cad_ids,detections,img_array.shape[:2])
    for cad_id,det_idx in matches:
        print(f"Detection {det_idx} ↔ CAD panel {cad_id}")

    # Visualize
    vis = img_array.copy()
    for d in detections: cv2.polylines(vis,[d],True,(0,255,0),2)
    for p in projected_cad: cv2.polylines(vis,[p.astype(int)],True,(0,0,255),2)
    cv2.imshow("IR & CAD",vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ---------------- CLI ----------------
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="CAD registration from IR image")
    parser.add_argument("-i","--image", type=str, required=True, help="Path to IR image")
    parser.add_argument("-c","--cad", type=str, default=cad_map, help="Path to CAD GeoJSON")
    args = parser.parse_args()
    cad_map = args.cad
    main( args.image)