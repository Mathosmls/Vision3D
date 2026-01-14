import bpy
import os
import math
import random

# =========================
# PARAMÈTRES
# =========================
N_IMAGES = 10
OUTPUT_DIR = "/tmp/calib"
LEFT_DIR = os.path.join(OUTPUT_DIR, "left")
RIGHT_DIR = os.path.join(OUTPUT_DIR, "right")

os.makedirs(LEFT_DIR, exist_ok=True)
os.makedirs(RIGHT_DIR, exist_ok=True)

cam_left = bpy.data.objects["CameraL"]
cam_right = bpy.data.objects["CameraR"]
chessboard = bpy.data.objects["Chessboard"]

scene = bpy.context.scene

# =========================
# FONCTION RENDER
# =========================
def render_camera(camera, filepath):
    scene.camera = camera
    scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)

# =========================
# BOUCLE DE CAPTURE
# =========================
# Sauvegarde de la pose initiale
init_location = chessboard.location.copy()
init_rotation = chessboard.rotation_euler.copy()

for i in range(N_IMAGES):

    chessboard.location = (
        init_location.x + random.uniform(-0.3, 0.3),
        init_location.y + random.uniform(-0.2, 0.2),
        init_location.z + random.uniform(-0.2, 0.2),
    )

    chessboard.rotation_euler = (
        init_rotation.x + random.uniform(-math.radians(20), math.radians(20)),
        init_rotation.y + random.uniform(-math.radians(20), math.radians(20)),
        init_rotation.z + random.uniform(-math.radians(30), math.radians(30)),
    )

    filename = f"img_{i:04d}.png"
    render_camera(cam_left, os.path.join(LEFT_DIR, filename))
    render_camera(cam_right, os.path.join(RIGHT_DIR, filename))

# Restauration finale
chessboard.location = init_location
chessboard.rotation_euler = init_rotation


print("Done.")
