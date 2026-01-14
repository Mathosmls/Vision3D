
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import json

with open("calibration_params.json", "r") as f:
    calib = json.load(f)

K_l_new = np.array(calib["K_l_new"])
D_l_new = np.array(calib["D_l_new"])
K_r_new = np.array(calib["K_r_new"])
D_r_new = np.array(calib["D_r_new"])
R = np.array(calib["R"])
T = np.array(calib["T"]).reshape(3,1)
E = np.array(calib["E"])
F = np.array(calib["F"])


# Choisir la paire d'images pour la depth map (ex: première paire valide)
img_l = cv.imread("../img_calib_blender/test_l.png", cv.IMREAD_GRAYSCALE)
img_r = cv.imread("../img_calib_blender/test_r.png", cv.IMREAD_GRAYSCALE)
h, w = img_l.shape

# -------------------------
# Rectification stéréo
# -------------------------
R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(
    K_l_new, D_l_new,
    K_r_new, D_r_new,
    (w, h),  # width, height
    R, T,
    flags=cv.CALIB_ZERO_DISPARITY,
    alpha=0
)

map1x, map1y = cv.initUndistortRectifyMap(K_l_new, D_l_new, R1, P1, (w,h), cv.CV_32FC1)
map2x, map2y = cv.initUndistortRectifyMap(K_r_new, D_r_new, R2, P2, (w,h), cv.CV_32FC1)

rectified_l = cv.remap(img_l, map1x, map1y, cv.INTER_LINEAR)
rectified_r = cv.remap(img_r, map2x, map2y, cv.INTER_LINEAR)

# -------------------------
# StereoSGBM
# -------------------------
min_disp = 0
num_disp = 128  # multiple de 16
block_size = 5

stereo = cv.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    P1=8*3*block_size**2,
    P2=32*3*block_size**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    preFilterCap=63,
    mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
)

disparity = stereo.compute(rectified_l, rectified_r).astype(np.float32) / 16.0

# -------------------------
# Filtre WLS pour lisser
# -------------------------
# wls_filter = cv.ximgproc.createDisparityWLSFilter(stereo)
# wls_filter.setLambda(8000)
# wls_filter.setSigmaColor(1.5)
# disparity_filtered = wls_filter.filter(disparity, rectified_l)
disparity_filtered = disparity 
# -------------------------
# Reprojection en 3D
# -------------------------
points_3d = cv.reprojectImageTo3D(disparity_filtered, Q)
mask = disparity_filtered > disparity_filtered.min()
depth_map = np.where(mask, points_3d[:,:,2], 0)  # Z = profondeur

# -------------------------
# Nettoyage des valeurs aberrantes
# -------------------------
max_depth = 200
min_depth = 0
valid_mask = (depth_map > min_depth) & (depth_map < max_depth)
depth_map_clean = np.where(valid_mask, depth_map, 0)

# Optionnel : median blur pour lisser
depth_map_clean = cv.medianBlur(depth_map_clean.astype(np.float32), 5)

# -------------------------
# Affichage
# -------------------------

fig, axes = plt.subplots(1, 3, figsize=(12, 6))

axes[0].imshow(rectified_l, cmap='gray')
axes[0].set_title("Caméra gauche rectifiée")
axes[0].axis('off')

axes[1].imshow(rectified_r, cmap='gray')
axes[1].set_title("Caméra droite rectifiée")
axes[1].axis('off')

axes[2].imshow(depth_map_clean, cmap='plasma')
axes[2].set_title("Depth map")
axes[2].axis('off')

plt.show()



plt.figure(figsize=(10,6))
plt.imshow(depth_map_clean, cmap='plasma')
plt.colorbar(label="Profondeur (unités objpoints)")
plt.title("Depth map SGBM + WLS")
plt.show()