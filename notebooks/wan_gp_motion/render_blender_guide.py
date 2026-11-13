"""Blender 4.x: blender --background --python render_blender_guide.py

124 frames / 24 fps / 864x480. Output is relative to the current working folder.
Camera locations are baked on every rendered frame; no legacy action.fcurves API.
"""
from pathlib import Path
import math
import bpy

FRAMES, FPS, WIDTH, HEIGHT = 124, 24, 864, 480
ORBIT_DEG, RADIUS, CAM_Z, TARGET_Z, LENS_MM = 30.0, 8.0, 1.6, 1.35, 50.0
OUT_DIR = Path.cwd()/'director_guide'
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene


def material(name, gray):
    result = bpy.data.materials.new(name)
    result.use_nodes = True
    bsdf = result.node_tree.nodes['Principled BSDF']
    bsdf.inputs['Base Color'].default_value = (gray, gray, gray, 1)
    bsdf.inputs['Roughness'].default_value = 1.0
    if 'Specular IOR Level' in bsdf.inputs:
        bsdf.inputs['Specular IOR Level'].default_value = 0.0
    return result


cross_mat, ground_mat = material('Cross', .85), material('Ground', .30)
# One watertight extruded outline avoids coplanar overlapping cube faces at the joint.
# Same dimensions as the handoff: 2.4 m tall, 1.6 m wide, 0.32 m thick beams.
outline = [(-.16, 0), (.16, 0), (.16, 1.59), (.8, 1.59), (.8, 1.91), (.16, 1.91),
           (.16, 2.4), (-.16, 2.4), (-.16, 1.91), (-.8, 1.91), (-.8, 1.59), (-.16, 1.59)]
count = len(outline)
vertices = [(x, y, z) for y in (-.16, .16) for x, z in outline]
faces = [tuple(range(count)), tuple(range(2*count-1, count-1, -1))]
faces += [((i+1) % count, i, i+count, (i+1) % count+count) for i in range(count)]
mesh = bpy.data.meshes.new('CrossSolid')
mesh.from_pydata(vertices, [], faces)
mesh.update()
obj = bpy.data.objects.new('Cross', mesh)
scene.collection.objects.link(obj)
obj.data.materials.append(cross_mat)
bpy.ops.mesh.primitive_plane_add(size=60)
bpy.context.object.data.materials.append(ground_mat)
world = bpy.data.worlds.new('GuideWorld')
world.use_nodes = True
world.node_tree.nodes['Background'].inputs[0].default_value = (.55, .55, .55, 1)
world.node_tree.nodes['Background'].inputs[1].default_value = 1.0
scene.world = world
camera_data = bpy.data.cameras.new('GuideCam')
camera_data.lens, camera_data.sensor_width = LENS_MM, 36
camera_data.dof.use_dof = False
camera = bpy.data.objects.new('GuideCam', camera_data)
scene.collection.objects.link(camera)
scene.camera = camera
aim = bpy.data.objects.new('Aim', None)
aim.location = (0, 0, TARGET_Z)
scene.collection.objects.link(aim)
track = camera.constraints.new('TRACK_TO')
track.target, track.track_axis, track.up_axis = aim, 'TRACK_NEGATIVE_Z', 'UP_Y'
for frame in range(1, FRAMES+1):
    angle = math.radians(-ORBIT_DEG/2+ORBIT_DEG*(frame-1)/(FRAMES-1))
    camera.location = (RADIUS*math.sin(angle), -RADIUS*math.cos(angle), CAM_Z)
    camera.keyframe_insert('location', frame=frame)
scene.frame_start, scene.frame_end = 1, FRAMES
scene.render.fps = FPS
scene.render.resolution_x, scene.render.resolution_y = WIDTH, HEIGHT
scene.render.resolution_percentage = 100
scene.render.engine = 'CYCLES'
scene.cycles.device, scene.cycles.samples = 'CPU', 16
scene.cycles.use_denoising = False
if hasattr(scene.render, 'use_motion_blur'):
    scene.render.use_motion_blur = False
scene.view_settings.view_transform = 'Standard'
scene.view_settings.exposure, scene.view_settings.gamma = 0, 1
scene.render.image_settings.file_format = 'FFMPEG'
scene.render.ffmpeg.format, scene.render.ffmpeg.codec = 'MPEG4', 'H264'
scene.render.ffmpeg.constant_rate_factor = 'HIGH'
scene.render.ffmpeg.audio_codec = 'NONE'
OUT_DIR.mkdir(parents=True, exist_ok=True)
scene.render.filepath = str(OUT_DIR/'JJP_orbit_124_')
print(f'GUIDE: {WIDTH}x{HEIGHT}, {FPS}fps, {FRAMES} frames, {ORBIT_DEG} degrees; locked {LENS_MM}mm')
bpy.ops.render.render(animation=True)
print('DONE:', OUT_DIR)
