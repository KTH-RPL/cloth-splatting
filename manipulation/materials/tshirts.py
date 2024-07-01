import bpy
from synthetic_cloth_data.materials import RGBAColor
from synthetic_cloth_data.materials.common import (
    ImageOnTextureConfig,
    add_image_to_material_base_color,
    create_evenly_colored_material,
    modify_bsdf_to_cloth,
)


def create_evenly_colored_tshirts_material(color: RGBAColor) -> bpy.types.Material:
    material = create_evenly_colored_material(color)
    return material


if __name__ == "__main__":
    import pathlib

    # Delete the default cube and add a plane with the dish towel material
    bpy.ops.object.delete()
    bpy.ops.mesh.primitive_plane_add()
    plane = bpy.context.object
    red = (1.0, 0.0, 0.0, 1.0)
    material = create_evenly_colored_tshirts_material(red)
    material = modify_bsdf_to_cloth(material)
    path = str(pathlib.Path(__file__).parent / "test.jpg")
    material = add_image_to_material_base_color(material, path, ImageOnTextureConfig())
    plane.data.materials.append(material)
