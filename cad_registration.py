#!/usr/bin/env python3
import os
import math
import json
import numpy as np
import cv2
from shapely.geometry import shape, Polygon, MultiPolygon
from scipy.optimize import linear_sum_assignment
from PIL import Image
import piexif
import argparse
import pyproj
import georef_new

# ---------------- Default Parameters ----------------
CAD_MAP = "section_1_ir_cad (1).geojson"

# ---------------- Helper Functions ----------------
def load_cad_polygons(geojson_path):
    """Load polygons from a CAD GeoJSON file."""
    with open(geojson_path, 'r') as f:
        cad_data = json.load(f)
    polygons, ids = [], []
    for i, feature in enumerate(cad_data.get('features', [])):
        geom = shape(feature['geometry'])
        if geom.is_empty:
            continue
        if isinstance(geom, Polygon):
            polygons.append(np.array(geom.exterior.coords))
            ids.append(feature.get("id", i))
        elif isinstance(geom, MultiPolygon):
            for j, poly in enumerate(geom.geoms):
                polygons.append(np.array(poly.exterior.coords))
                ids.append(f"{feature.get('id', i)}_{j}")
    return polygons, ids

def detect_coord_system(polygons):
    """Detect if coordinates are lon/lat or meters."""
    if len(polygons) == 0:
        return 'meters'
    pts = np.vstack([p[:,:2].reshape(-1,2) for p in polygons if p.size > 0])
    xs, ys = pts[:,0], pts[:,1]
    if np.all((xs >= -180) & (xs <= 180)) and np.all((ys >= -90) & (ys <= 90)):
        return 'lonlat'
    if np.all((xs >= -90) & (xs <= 90)) and np.all((ys >= -180) & (ys <= 180)):
        return 'latlon_swapped'
    if np.max(np.abs(pts)) > 1000:
        return 'meters'
    return 'lonlat'

def compute_utm_proj_for_lonlat(lon, lat):
    zone = int(math.floor((lon + 180) / 6) + 1)
    south = lat < 0
    proj = pyproj.Proj(proj="utm", zone=zone, ellps="WGS84", south=south)
    return proj, zone, south

def geojson_to_local(polygons, ref_lat, ref_lon, coords_mode='auto'):
    """Convert CAD polygons to local coordinates."""
    detected = detect_coord_system(polygons) if coords_mode=='auto' else coords_mode
    if detected == 'meters':
        return [p[:,:2].astype(float) for p in polygons], None, None

    proj, _, _ = compute_utm_proj_for_lonlat(ref_lon, ref_lat)
    ref_x, ref_y = proj(ref_lon, ref_lat)
    local_polys = []
    for p in polygons:
        coords = []
        arr = np.array(p)
        for coord in arr:
            lon, lat = (coord[0], coord[1]) if detected=='lonlat' else (coord[1], coord[0])
            x, y = proj(lon, lat)
            coords.append([x-ref_x, y-ref_y])
        local_polys.append(np.array(coords, dtype=float))
    return local_polys, None, None

# ---------------- Projection Functions ----------------
def project_points_world_to_image(points_xyz, K, R, t):
    """Project 3D points to 2D image coordinates."""
    points_xyz = np.atleast_2d(points_xyz)
    if points_xyz.shape[1] == 2:
        points_xyz = np.hstack([points_xyz, np.zeros((points_xyz.shape[0],1))])
    proj = K @ (R @ points_xyz.T + t)
    zs = proj[2,:]
    uv = np.full((points_xyz.shape[0],2), np.nan, dtype=np.float32)
    mask = zs > 1e-6
    uv[mask,0] = proj[0,mask]/zs[mask]
    uv[mask,1] = proj[1,mask]/zs[mask]
    return uv

def project_polygons(cad_polygons_world, K, R, t, panel_height=2.0):
    projected = []
    for poly in cad_polygons_world:
        if poly.size == 0:
            projected.append(np.empty((0,2)))
            continue
        poly_xyz = np.hstack([poly, np.full((poly.shape[0],1), panel_height)]) if poly.shape[1]==2 else poly
        projected.append(project_points_world_to_image(poly_xyz, K, R, t))
    return projected

# ---------------- Bounding Box & Matching ----------------
def bbox_from_poly(poly):
    x_min, y_min = np.nanmin(poly[:,0]), np.nanmin(poly[:,1])
    x_max, y_max = np.nanmax(poly[:,0]), np.nanmax(poly[:,1])
    return x_min, y_min, x_max, y_max

def iou_bbox(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2-ix1)*(iy2-iy1)
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)

def match_detections_to_cad(projected_cad, cad_ids, detections, img_shape):
    proj_bboxes = [bbox_from_poly(p) for p in projected_cad]
    det_bboxes = [bbox_from_poly(np.array(d)) for d in detections]
    if not proj_bboxes or not det_bboxes:
        return []
    cost = np.full((len(proj_bboxes), len(det_bboxes)), 1e6, dtype=np.float32)
    diag = np.hypot(*img_shape[::-1])
    for i, cad_bb in enumerate(proj_bboxes):
        cx_c, cy_c = (cad_bb[0]+cad_bb[2])/2, (cad_bb[1]+cad_bb[3])/2
        for j, det_bb in enumerate(det_bboxes):
            cx_d, cy_d = (det_bb[0]+det_bb[2])/2, (det_bb[1]+det_bb[3])/2
            dist = np.hypot(cx_c-cx_d, cy_c-cy_d)
            cost[i,j] = 0.7*(dist/diag) + 0.3*(1-iou_bbox(cad_bb, det_bb))
    row, col = linear_sum_assignment(cost)
    return [(cad_ids[r], c) for r,c in zip(row,col) if cost[r,c] < 0.5]

# ---------------- Image Utilities ----------------
def load_ir_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = Image.open(image_path)
    try:
        exif_dict = piexif.load(img.info.get('exif', image_path))
    except:
        exif_dict = {}
    img_array = np.array(img)
    if img_array.dtype==np.uint16:
        img_array = (img_array/256).astype(np.uint8)
    if img_array.ndim==2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    return img_array, exif_dict

def draw_polygons_on_image(img, polygons, min_area=100):
    h, w = img.shape[:2]
    vis = img.copy()
    for p in polygons:
        if p.size==0: continue
        valid = ~np.isnan(p[:,0]) & ~np.isnan(p[:,1])
        if not np.any(valid): continue
        pts = p.copy()
        if np.any(~valid):
            centroid = np.nanmean(pts[valid], axis=0)
            pts[~valid] = centroid
        bbox = bbox_from_poly(pts)
        area = max((bbox[2]-bbox[0])*(bbox[3]-bbox[1]), 0.0)
        if area < min_area: continue
        pts_int = np.int32(np.round(pts))
        cv2.polylines(vis, [pts_int], True, (0,0,255), max(1,int(round(2.0*(w/1000.0)))))
    return vis

# ---------------- Main Pipeline ----------------
def main_pipeline(ir_image_path, cad_geojson_path, coords_mode='auto', panel_height=2.0, min_visible_pix_area=100, coords_scale=1.0):
    img_array, exif_dict = load_ir_image(ir_image_path)

    drone_gps_corr, drone_gps_raw, drone_pose = georef_new.get_corrected_drone_gps(exif_dict)
    lat, lon = drone_gps_corr
    alt = drone_gps_raw[2] if len(drone_gps_raw)>2 else 0.0
    yaw, pitch, roll = drone_pose

    cad_polygons, cad_ids = load_cad_polygons(cad_geojson_path)
    if not cad_polygons:
        print("No polygons found in CAD GeoJSON.")
        return

    cad_polygons_local, _, _ = geojson_to_local(cad_polygons, lat, lon, coords_mode)
    if coords_scale != 1.0:
        cad_polygons_local = [p*coords_scale for p in cad_polygons_local]

    R_c2w = georef_new.rotation_matrix_from_rpy(roll, pitch, yaw)
    R = R_c2w.T
    t = -R @ np.array([0,0,alt]).reshape(3,1)
    if not hasattr(georef_new, 'K'):
        print("Error: georef_new.K not set.")
        return
    K = georef_new.K

    projected_cad = project_polygons(cad_polygons_local, K, R, t, panel_height)
    vis = draw_polygons_on_image(img_array, projected_cad, min_area=min_visible_pix_area)

    cv2.imshow("IR & CAD Projection", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ---------------- CLI ----------------
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="CAD Projection (Auto Coordinate Detection)")
    parser.add_argument("-i","--image", required=True, help="Path to IR image")
    parser.add_argument("--coords-scale", type=float, default=1.0, help="Manual scaling for CAD coordinates")
    args = parser.parse_args()

    main_pipeline(args.image, CAD_MAP, coords_mode='auto', panel_height=2.0,
                  min_visible_pix_area=100.0, coords_scale=args.coords_scale)
