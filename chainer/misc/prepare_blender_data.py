import math

import bpy
import mathutils


def clear():
    bpy.ops.wm.open_mainfile(filepath='./tests/data/clean.blend')


def setup(image_size):
    context = bpy.context
    context.scene.render.resolution_x = image_size
    context.scene.render.resolution_y = image_size
    context.scene.render.resolution_percentage = 100
    context.scene.render.use_antialiasing = False
    context.scene.render.alpha_mode = 'SKY'
    bpy.context.scene.render.image_settings.color_mode = 'RGB'
    context.scene.world.horizon_color = (255, 255, 255)

    # camera
    camera = bpy.data.cameras.values()[0]
    camera.sensor_width = 2
    camera.sensor_height = 2
    camera.lens = 1.732

    # lighting
    light = bpy.data.objects['Lamp']
    light.data.energy = 1
    context.scene.world.light_settings.use_environment_light = True
    context.scene.world.light_settings.environment_energy = 0.5
    context.scene.world.light_settings.environment_color = 'PLAIN'


def load_obj(filename):
    bpy.ops.import_scene.obj(
        filepath=filename, use_smooth_groups=False, use_split_objects=False, use_split_groups=False)
    object_id = len(bpy.data.objects) - 1
    obj = bpy.data.objects[object_id]
    bpy.context.scene.objects.active = obj

    # normalization
    v_min = []
    v_max = []
    for i in range(3):
        v_min.append(min([vertex.co[i] for vertex in obj.data.vertices]))
        v_max.append(max([vertex.co[i] for vertex in obj.data.vertices]))

    v_min = mathutils.Vector(v_min)
    v_max = mathutils.Vector(v_max)
    scale = max(v_max - v_min)
    v_shift = (v_max - v_min) / 2 / scale

    for v in obj.data.vertices:
        v.co -= v_min
        v.co /= scale
        v.co -= v_shift
        v.co *= 2


def set_camera_location(elevation, azimuth, distance):
    # from https://blender.stackexchange.com/questions/18530/
    x = 1 * math.cos(math.radians(-azimuth)) * math.cos(math.radians(elevation)) * distance
    y = 1 * math.sin(math.radians(-azimuth)) * math.cos(math.radians(elevation)) * distance
    z = 1 * math.sin(math.radians(elevation)) * distance
    camera = bpy.data.objects["Camera"]
    camera.location = x, y, z

    direction = - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()


def render(filename):
    bpy.context.scene.render.filepath = filename
    bpy.ops.render.render(write_still=True)


def run():
    image_size = 256
    distance = 2.732
    azimuth = 90
    elevation = 0

    clear()
    setup(image_size)
    load_obj('./tests/data/teapot.obj')
    set_camera_location(elevation, azimuth, distance)
    render('./tests/data/teapot_blender.png')


if __name__ == '__main__':
    run()
