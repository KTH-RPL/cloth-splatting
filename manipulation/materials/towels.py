import bpy



def _add_white_stripes_on_black_nodes(
    node_tree: bpy.types.ShaderNodeTree,
    input_socket: bpy.types.NodeSocket,
    amount_of_stripes: int,
    stripe_width: float,
) -> bpy.types.NodeSocket:
    """
    Add nodes to a node tree to create a pattern with white stripes on a black background.
    Args:
        node_tree: The matieral node tree to add the nodes to.
        input_socket: This input should be a linear 0->1 range, e.g. the X component of a UV coordinate.
        amount_of_stripes: The amount of white stripes on the black background.
        stripe_width: The relative width of the white stripes, in range 0.0 to 1.0. 0.0 is solid black, 1.0 is solid white.
    Returns:
        The Color output socket that contains the striped pattern.
    """
    nodes = node_tree.nodes
    links = node_tree.links

    # Add nodes
    # Math node set to multiply by 10
    multiply = nodes.new(type="ShaderNodeMath")
    multiply.operation = "MULTIPLY"
    links.new(input_socket, multiply.inputs[0])
    multiply.inputs[1].default_value = amount_of_stripes

    # Math node set to fraction
    fraction = nodes.new(type="ShaderNodeMath")
    fraction.operation = "FRACT"
    links.new(multiply.outputs["Value"], fraction.inputs[0])

    # Math node set to compare to 0.5, with epsilon in range 0.0 to 0.5 to control the stripe width
    compare = nodes.new(type="ShaderNodeMath")
    compare.operation = "COMPARE"
    links.new(fraction.outputs["Value"], compare.inputs[0])
    compare.inputs[1].default_value = 0.5
    compare.inputs[2].default_value = 0.5 * stripe_width
    output_socket = compare.outputs["Value"]

    return output_socket



def create_gridded_dish_towel_material(
    amount_of_vertical_stripes: int,
    amount_of_horizontal_stripes: int,
    vertical_stripe_width: float,
    horizontal_stripe_width: float,
    vertical_stripe_color: tuple[float, float, float, float],
    horizontal_stripe_color: tuple[float, float, float, float],
    intersection_color: tuple[float, float, float, float],
    background_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> bpy.types.Material:
    """Creates a dish towel material with horizontal and vertical stripes.
    Args:
        amount_of_vertical_stripes: The amount of vertical stripes.
        amount_of_horizontal_stripes: The amount of horizontal stripes.
        vertical_stripe_width: The relative width of the vertical stripes.
        horizontal_stripe_width: The relative width of the horizontal stripes.
        vertical_stripe_color: The color of the vertical stripes.
        horizontal_stripe_color: The color of the horizontal stripes.
        intersection_color: The color at the intersections of the vertical and horizontal stripes.
        background_color: The color of the background. Defaults to white.
    Returns:
        bpy.types.Material: The created material.
    """
    material = bpy.data.materials.new(name="Gridded Dish Towel")
    material.use_nodes = True

    node_tree = material.node_tree
    nodes = node_tree.nodes
    links = node_tree.links

    texture_coordinates = nodes.new(type="ShaderNodeTexCoord")
    separate_xyz = nodes.new(type="ShaderNodeSeparateXYZ")

    links.new(texture_coordinates.outputs["UV"], separate_xyz.inputs["Vector"])

    vertical_stripes = _add_white_stripes_on_black_nodes(
        node_tree, separate_xyz.outputs["X"], amount_of_vertical_stripes, vertical_stripe_width
    )
    horizontal_stripes = _add_white_stripes_on_black_nodes(
        node_tree, separate_xyz.outputs["Y"], amount_of_horizontal_stripes, horizontal_stripe_width
    )

    # Find where the stripes intersect by multiplying the vertical and horizontal stripes
    multiply = nodes.new(type="ShaderNodeMath")
    multiply.operation = "MULTIPLY"
    links.new(vertical_stripes, multiply.inputs[0])
    links.new(horizontal_stripes, multiply.inputs[1])
    stripes = multiply.outputs["Value"]

    # Now create the colors by using a Mix node
    mix = nodes.new(type="ShaderNodeMix")
    mix.data_type = "RGBA"
    links.new(vertical_stripes, mix.inputs[0])
    mix.inputs[6].default_value = background_color
    mix.inputs[7].default_value = vertical_stripe_color

    # Add another Mix node for the horizontal stripes
    mix2 = nodes.new(type="ShaderNodeMix")
    mix2.data_type = "RGBA"
    links.new(horizontal_stripes, mix2.inputs[0])
    links.new(mix.outputs[2], mix2.inputs[6])
    mix2.inputs[7].default_value = horizontal_stripe_color

    # Add a third mix node for the intersection of the stripes
    mix3 = nodes.new(type="ShaderNodeMix")
    mix3.data_type = "RGBA"
    links.new(stripes, mix3.inputs[0])
    links.new(mix2.outputs[2], mix3.inputs[6])
    mix3.inputs[7].default_value = intersection_color
    colored_stripes = mix3.outputs[2]

    links.new(colored_stripes, nodes["Principled BSDF"].inputs["Base Color"])
    return material


def create_striped_material(
    amount_of_stripes: int,
    stripe_width: float,
    stripe_color: tuple[float, float, float, float],
    background_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    vertical: bool = True,
) -> bpy.types.Material:
    """
    Create a striped dish towel material.
    Args:
        amount_of_stripes: The amount of stripes.
        stripe_width: The relative width of the stripes, 0.0 means only background color, 1.0 means only stripe color.
        stripe_color: The color of the stripes.
        background_color: The color of the background. Defaults to white.
        vertical: If True, the stripes run vertically, else horizontally.
    Returns:
        The created material.
    """
    material = bpy.data.materials.new(name="Striped Dish Towel")
    material.use_nodes = True

    node_tree = material.node_tree
    nodes = node_tree.nodes
    links = node_tree.links

    # First set up the texture coordinate node for access to the UVs
    texture_coordinates = nodes.new(type="ShaderNodeTexCoord")
    separate_xyz = nodes.new(type="ShaderNodeSeparateXYZ")

    # Connect the texture coordinate node to the separate XYZ node
    links.new(texture_coordinates.outputs["UV"], separate_xyz.inputs["Vector"])

    output_name = "X" if vertical else "Y"
    stripes = _add_white_stripes_on_black_nodes(
        node_tree, separate_xyz.outputs[output_name], amount_of_stripes, stripe_width
    )

    # There are several ways to turn a black and white pattern into a colored pattern.
    # Here we use a Mix node to mix the stripe color with the background color.
    # TODO: consider a better way of specifying inputs and outputs than using indices
    # We can't use the string names because they are not unique.
    mix = nodes.new(type="ShaderNodeMix")
    mix.data_type = "RGBA"
    links.new(stripes, mix.inputs[0])
    mix.inputs[6].default_value = background_color
    mix.inputs[7].default_value = stripe_color
    colored_stripes = mix.outputs[2]

    links.new(colored_stripes, nodes["Principled BSDF"].inputs["Base Color"])
    return material


if __name__ == "__main__":
    # Delete the default cube and add a plane with the dish towel material
    bpy.ops.object.delete()
    bpy.ops.mesh.primitive_plane_add()
    plane = bpy.context.object
    red = (1.0, 0.0, 0.0, 1.0)
    material = create_striped_material(3, 0.5, red)
    material = modify_bsdf_to_cloth(material)
    plane.data.materials.append(material)

    # Add a second plane with the gridded dish towel material
    bpy.ops.mesh.primitive_plane_add()
    plane_gridded = bpy.context.object
    plane_gridded.location = (2.1, 0.0, 0.0)
    pale_blue = (0.5, 0.5, 1.0, 1.0)
    pale_yellow = (1.0, 1.0, 0.5, 1.0)
    green = (0.0, 1.0, 0.0, 1.0)
    material = create_gridded_dish_towel_material(5, 5, 0.2, 0.2, pale_blue, pale_yellow, green)
    # material = create_evenly_colored_material(green)
    material = modify_bsdf_to_cloth(material)
    plane_gridded.data.materials.append(material)
