import bpy
import os

# =========================
# Paramètres
# =========================
OUTPUT_DIR = "/tmp/stereo_ball_frames"
LEFT_DIR = os.path.join(OUTPUT_DIR, "left")
RIGHT_DIR = os.path.join(OUTPUT_DIR, "right")
os.makedirs(LEFT_DIR, exist_ok=True)
os.makedirs(RIGHT_DIR, exist_ok=True)

cam_left = bpy.data.objects["CameraL"]
cam_right = bpy.data.objects["CameraR"]

scene = bpy.context.scene
fps = scene.render.fps  # framerate de la timeline
duration_seconds = 2    # nombre de secondes à rendre
num_frames = int(fps * duration_seconds)

start_frame = scene.frame_start
end_frame = min(scene.frame_start + 8 - 1, scene.frame_end)

# Résolution
scene.render.resolution_x = 1280
scene.render.resolution_y = 720
scene.render.resolution_percentage = 100

# Format image
scene.render.image_settings.file_format = 'PNG'

# =========================
# Boucle de rendu frame par frame
# =========================
for f in range(start_frame, end_frame + 1):
    scene.frame_set(f)  # positionne l'animation à la frame f
    
    # Caméra gauche
    scene.camera = cam_left
    scene.render.filepath = os.path.join(LEFT_DIR, f"img_{f:04d}.png")
    bpy.ops.render.render(write_still=True)
    
    # Caméra droite
    scene.camera = cam_right
    scene.render.filepath = os.path.join(RIGHT_DIR, f"img_{f:04d}.png")
    bpy.ops.render.render(write_still=True)
    
    print(f"Frame {f} rendue pour les deux caméras")

print(f"Rendu des {duration_seconds} premières secondes terminé !")
