import cv2
import numpy as np
from PIL import Image

# --- Load image ---
image_path = r"C:\Users\Micha\Documents\Schriftverkehr\Uni\Praktikum Kopenhagen\Documents internship\img_ir_gps_big\15-39-40-876-radiometric.tiff"
img = Image.open(image_path)
img_array = np.array(img)

if len(img_array.shape) == 3:
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
else:
    gray = img_array

# --- Enhance contrast ---
gray_eq = cv2.equalizeHist(gray)

# --- Gaussian blur (optional noise reduction) ---
blur = cv2.GaussianBlur(gray_eq, (3,3), 0)

# --- Edge detection (Canny) ---
edges = cv2.Canny(blur, threshold1=50, threshold2=150)

# --- Morphological closing to connect gaps ---
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# --- Show results ---
cv2.imshow("Grayscale Equalized", gray_eq)
cv2.imshow("Canny Edges", edges)
cv2.imshow("Closed Edges", closed)
cv2.waitKey(0)
cv2.destroyAllWindows()
