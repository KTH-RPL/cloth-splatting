from pathlib import Path

import bpy
import numpy as np
from synthetic_cloth_data.materials.ambientCGFabrics.download_ambientcg_fabrics import AMBIENTCG_CLOTH_MATERIALS_DIR
from synthetic_cloth_data.materials.common import _add_noise_texture_to_bsdf_normals


def find_normal_map_of_ambientcg_material(material_dir: Path):
    # each material dir has a bunch of maps, find the normal map which contains normalGL in its name
    material_maps = [str(p) for p in material_dir.glob("*.jpg")]
    normal_map = None
    for map in material_maps:
        if "normalgl" in map.lower():
            normal_map = map
            break
    return normal_map


def add_random_ambientcg_cloth_normal_to_object(material: bpy.types.Material):
    """add the normal maps of an ambientcg cloth material to a blender material.
    Intended use is to later modify the base color of the material but to mimick the low-level
    details of a cloth material.
    """
    material_dir = np.random.choice(list(AMBIENTCG_CLOTH_MATERIALS_DIR.glob("*")))
    print(material_dir)
    normal_map = find_normal_map_of_ambientcg_material(material_dir)
    if normal_map is None:
        raise ValueError("No normal map found in material dir")

    nodes = material.node_tree.nodes
    links = material.node_tree.links

    # First set up the texture coordinate node for access to the UVs
    texture_coordinates = nodes.new(type="ShaderNodeTexCoord")
    texture_mapping_node = nodes.new(type="ShaderNodeMapping")

    # Connect the texture coordinate node to the separate XYZ node
    links.new(texture_coordinates.outputs["UV"], texture_mapping_node.inputs["Vector"])

    # create blender image node, link to UV coords
    image_node = material.node_tree.nodes.new("ShaderNodeTexImage")
    image_node.image = bpy.data.images.load(normal_map)
    links.new(texture_mapping_node.outputs["Vector"], image_node.inputs["Vector"])
    links.new(image_node.outputs["Color"], material.node_tree.nodes["Principled BSDF"].inputs["Normal"])


if __name__ == "__main__":
    add_random_ambientcg_cloth_normal_to_object(bpy.context.object.active_material)
    _add_noise_texture_to_bsdf_normals(bpy.context.object.active_material, 20, 0.001)
