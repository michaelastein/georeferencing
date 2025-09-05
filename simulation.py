import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Slider
from matplotlib.patches import Patch

# Rotationsmatrix aus Euler-Winkeln
def rotation_matrix(pitch, yaw, roll):
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)
    roll = np.radians(roll)
    
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch), np.cos(pitch)]])
    
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                   [0, 1, 0],
                   [-np.sin(yaw), 0, np.cos(yaw)]])
    
    Rz = np.array([[np.cos(roll), -np.sin(roll), 0],
                   [np.sin(roll), np.cos(roll), 0],
                   [0, 0, 1]])
    
    return Rz @ Ry @ Rx

# Projektion der Kamera-FOV auf Boden mit festem 640x512 Verhältnis
def project_fov(pitch, yaw, roll, fov=60,
                cam_height=1.5, image_offset=0.5):
    """
    cam_height: Kameraposition über Boden
    image_offset: Abstand der Bildfläche über Kamera
    fov: vertikaler FOV in Grad
    Bildfläche im Verhältnis 640x512
    """
    aspect = 640 / 512  # 1.25
    R = rotation_matrix(pitch, yaw, roll)
    
    fov_rad = np.radians(fov/2)
    y = np.tan(fov_rad)
    x = y * aspect
    
    # Eckpunkte in Kamerakoordinaten (rechteckig)
    corners = np.array([
        [-x, -y, -1],
        [ x, -y, -1],
        [ x,  y, -1],
        [-x,  y, -1]
    ])
    
    cam_pos = np.array([0, 0, cam_height])
    corners_world = corners @ R.T + cam_pos + np.array([0,0,image_offset])
    
    # Projektion auf Boden
    dir_vectors = corners_world - cam_pos
    epsilon = 1e-6
    t = -cam_pos[2] / np.where(dir_vectors[:,2]==0, epsilon, dir_vectors[:,2])
    proj = cam_pos + dir_vectors * t[:, np.newaxis]
    
    return corners_world, proj, cam_pos

# Update-Funktion für Slider
def update(val):
    ax.cla()
    
    pitch = slider_pitch.val
    yaw   = slider_yaw.val
    roll  = slider_roll.val
    fov   = slider_fov.val
    
    corners_world, proj, cam_pos = project_fov(
        pitch, yaw, roll, fov=fov, cam_height=1.5, image_offset=0.5)
    
    # Bildfläche als Fläche
    ax.add_collection3d(Poly3DCollection([corners_world], facecolors='blue', alpha=0.5))
    
    # Projektion auf Boden als Fläche
    ax.add_collection3d(Poly3DCollection([proj], facecolors='green', alpha=0.5))
    
    # Bodenebene
    floor = np.array([[-3,-3,0],[3,-3,0],[3,3,0],[-3,3,0]])
    ax.add_collection3d(Poly3DCollection([floor], facecolors='lightgrey', alpha=0.3))
    
    ax.set_xlim(-3,3)
    ax.set_ylim(-3,3)
    ax.set_zlim(0,3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(30,30)
    
    # Dummy Handles für Legende
    blue_patch = Patch(color='blue', label='Image Plane')
    green_patch = Patch(color='green', label='Projected on floor')
    ax.legend(handles=[blue_patch, green_patch])
    
    fig.canvas.draw_idle()

# Plot Setup
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

axcolor = 'lightgoldenrodyellow'
ax_pitch = plt.axes([0.25, 0.01, 0.65, 0.03], facecolor=axcolor)
ax_yaw   = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
ax_roll  = plt.axes([0.25, 0.09, 0.65, 0.03], facecolor=axcolor)
ax_fov   = plt.axes([0.25, 0.13, 0.65, 0.03], facecolor=axcolor)

slider_pitch = Slider(ax_pitch, 'Roll', -90, 90, valinit=0)   # originally 'Pitch'
slider_yaw   = Slider(ax_yaw, 'Pitch', -90, 90, valinit=0)   # originally 'Yaw'
slider_roll  = Slider(ax_roll, 'Yaw', -180, 180, valinit=0)    # originally 'Roll'
slider_fov   = Slider(ax_fov, 'FOV', 10, 120, valinit=60)


slider_pitch.on_changed(update)
slider_yaw.on_changed(update)
slider_roll.on_changed(update)
slider_fov.on_changed(update)

update(0)
plt.show()
