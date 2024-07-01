import pathlib

import bpy
from synthetic_cloth_data.materials.common import (
    ImageOnTextureConfig,
    add_image_to_material_base_color,
    create_striped_material,
    modify_bsdf_to_cloth,
)

if __name__ == "__main__":
    # Delete the default cube and add a plane with the dish towel material
    bpy.ops.object.delete()

    import numpy as np

    vertices = [
        np.array([-0.25, 0.3, 0.0]),
        np.array([0.25, 0.3, 0.0]),
        np.array([0.34933467, -0.19003329, 0.0]),
        np.array([0.10431802, -0.23970062, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-0.10431802, -0.23970062, 0.0]),
        np.array([-0.34933467, -0.19003329, 0.0]),
    ]
    mesh = bpy.data.meshes.new("mesh")
    mesh.from_pydata(vertices, [], [[i for i in range(len(vertices))]])
    mesh.update()
    obj = bpy.data.objects.new("obj", mesh)
    bpy.context.collection.objects.link(obj)
    red = (1.0, 0.0, 0.0, 1.0)
    blue = (0.0, 0.0, 1.0, 1.0)
    material = create_striped_material(3, 0.5, blue, red)
    material = modify_bsdf_to_cloth(material)
    path = str(pathlib.Path(__file__).parent / "test.jpg")
    material = add_image_to_material_base_color(material, path, ImageOnTextureConfig())
    obj.data.materials.append(material)

    # activate the object and enter edit mode
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")

    # unwrap UV for rendering
    bpy.ops.uv.unwrap(method="ANGLE_BASED", margin=0.001)

    # exit edit mode
    bpy.ops.object.mode_set(mode="OBJECT")
