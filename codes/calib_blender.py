import numpy as np
import cv2 as cv
import glob
import os

def load_images_from_folder(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        img = cv.imread(os.path.join(folder,filename),cv.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images


def chessboardCorners(images,l=6,L=6): 
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((l*L,3), np.float32)
    objp[:,:2] = np.mgrid[0:l,0:L].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    valid_idx = [] 

    for idx,img in enumerate(images):
        # img = cv.imread(fname)
        # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(
        img,
        (l, L),
        flags=cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE
)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv.cornerSubPix(img,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            valid_idx.append(idx)

            # # Draw and display the corners
            # cv.drawChessboardCorners(img, (l,L), corners2, ret)
            # cv.imshow('img', img)
            # cv.waitKey(500)
    return objpoints, imgpoints,valid_idx

images_l = load_images_from_folder("../img_calib_blender/left")
images_r = load_images_from_folder("../img_calib_blender/right")
print(images_l[0].shape)
# for elem in images_l :
#     cv.imshow('img', elem)
#     cv.waitKey(500)
# for elem in images_r :
#     cv.imshow('img', elem)
#     cv.waitKey(500)

objpoints_l, imgpoints_l, valid_idx_l = chessboardCorners(images_l)
objpoints_r, imgpoints_r, valid_idx_r = chessboardCorners(images_r)
cv.destroyAllWindows()


# Trouver les indices valides communs (paires gauche-droite)
valid_pairs = list(set(valid_idx_l) & set(valid_idx_r))
valid_pairs.sort()  # tri pour garder l'ordre

# Filtrer les points
objpoints_l = [objpoints_l[valid_idx_l.index(i)] for i in valid_pairs]
imgpoints_l = [imgpoints_l[valid_idx_l.index(i)] for i in valid_pairs]
objpoints_r = [objpoints_r[valid_idx_r.index(i)] for i in valid_pairs]
imgpoints_r = [imgpoints_r[valid_idx_r.index(i)] for i in valid_pairs]

print(f"Nombre de paires valides : {len(valid_pairs)}")

ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv.calibrateCamera(objpoints_l, imgpoints_l, images_l[0].shape[::-1], None, None)
ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv.calibrateCamera(objpoints_r, imgpoints_r, images_l[0].shape[::-1], None, None)

print("fx =", mtx_l[0,0], "fy =", mtx_l[1,1])

criteria_stereo = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 100, 1e-5)

flags = cv.CALIB_FIX_INTRINSIC  # ne pas toucher K1/K2, D1/D2

ret_stereo, K_l_new, D_l_new, K_r_new, D_r_new, R, T, E, F = cv.stereoCalibrate(
    objpoints_l,
    imgpoints_l,
    imgpoints_r,
    mtx_l,
    dist_l,
    mtx_r,
    dist_r,
    images_l[0].shape[::-1],
    criteria=criteria_stereo,
    flags=flags
)

print("RMS error:", ret_stereo)
print("Rotation matrix R:\n", R)
print("Translation vector T:\n", T)


import numpy as np
import cv2 as cv
import glob
import os

def load_images_from_folder(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        img = cv.imread(os.path.join(folder,filename),cv.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images


def chessboardCorners(images,l=6,L=6): 
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((l*L,3), np.float32)
    objp[:,:2] = np.mgrid[0:l,0:L].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    valid_idx = [] 

    for idx,img in enumerate(images):
        # img = cv.imread(fname)
        # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(
        img,
        (l, L),
        flags=cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE
)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv.cornerSubPix(img,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            valid_idx.append(idx)

            # # Draw and display the corners
            # cv.drawChessboardCorners(img, (l,L), corners2, ret)
            # cv.imshow('img', img)
            # cv.waitKey(500)
    return objpoints, imgpoints,valid_idx

images_l = load_images_from_folder("../img_calib_blender/left")
images_r = load_images_from_folder("../img_calib_blender/right")
print(images_l[0].shape)
# for elem in images_l :
#     cv.imshow('img', elem)
#     cv.waitKey(500)
# for elem in images_r :
#     cv.imshow('img', elem)
#     cv.waitKey(500)

objpoints_l, imgpoints_l, valid_idx_l = chessboardCorners(images_l)
objpoints_r, imgpoints_r, valid_idx_r = chessboardCorners(images_r)
cv.destroyAllWindows()


# Trouver les indices valides communs (paires gauche-droite)
valid_pairs = list(set(valid_idx_l) & set(valid_idx_r))
valid_pairs.sort()  # tri pour garder l'ordre

# Filtrer les points
objpoints_l = [objpoints_l[valid_idx_l.index(i)] for i in valid_pairs]
imgpoints_l = [imgpoints_l[valid_idx_l.index(i)] for i in valid_pairs]
objpoints_r = [objpoints_r[valid_idx_r.index(i)] for i in valid_pairs]
imgpoints_r = [imgpoints_r[valid_idx_r.index(i)] for i in valid_pairs]

print(f"Nombre de paires valides : {len(valid_pairs)}")

ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv.calibrateCamera(objpoints_l, imgpoints_l, images_l[0].shape[::-1], None, None)
ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv.calibrateCamera(objpoints_r, imgpoints_r, images_l[0].shape[::-1], None, None)

print("fx =", mtx_l[0,0], "fy =", mtx_l[1,1])

criteria_stereo = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 100, 1e-5)

flags = cv.CALIB_FIX_INTRINSIC  # ne pas toucher K1/K2, D1/D2

ret_stereo, K_l_new, D_l_new, K_r_new, D_r_new, R, T, E, F = cv.stereoCalibrate(
    objpoints_l,
    imgpoints_l,
    imgpoints_r,
    mtx_l,
    dist_l,
    mtx_r,
    dist_r,
    images_l[0].shape[::-1],
    criteria=criteria_stereo,
    flags=flags
)

print("RMS error:", ret_stereo)
print("Rotation matrix R:\n", R)
print("Translation vector T:\n", T)


import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

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
# Calcul de la disparity
# -------------------------
num_disparities = 64  # multiple de 16
block_size = 15       # taille du patch
stereo = cv.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
disparity = stereo.compute(rectified_l, rectified_r).astype(np.float32) / 16.0

# -------------------------
# Reprojection en 3D
# -------------------------
points_3d = cv.reprojectImageTo3D(disparity, Q)
mask = disparity > disparity.min()
depth_map = np.where(mask, points_3d[:,:,2], 0)  # Z = profondeur

max_depth = 2000
min_depth = 0
valid_mask = (depth_map > min_depth) & (depth_map < max_depth)
depth_clean = np.where(valid_mask, depth_map, 0)

# Affichage
plt.figure(figsize=(10,6))
plt.imshow(depth_clean, cmap='plasma')
plt.colorbar(label="Profondeur (unités objpoints)")
plt.title("Depth map nettoyée")
plt.show()