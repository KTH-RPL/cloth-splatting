OptimizationParams = dict(

    iterations=12_000,

    mesh_type='multi',

    densification_interval=200,
    densify_from_iter=200,
    densify_until_iter=6500,
    #
    densify_grad_threshold_fine_init=0.001,
    densify_grad_threshold_after=0.001,
    opacity_reset_interval=550,
    #
    pruning_from_iter=200,
    pruningy_until_iter=6500,
    pruning_interval=200,
    percent_dense=0.01,
    opacity_threshold_fine_init=0.005,
    opacity_threshold_fine_after=0.005,

    gaussian_init_factor=2,

    no_coarse=True,
    white_background=True,

    scaling_lr=0.005,
    rotation_lr=0.001,
    position_lr_init=0.000016,
    position_lr_final=0.00000016,
    position_lr_delay_mult=0.1,
    position_lr_max_steps=5500,
    feature_lr=0.00025,

    position_lr_static=0.0016,
    static_reconst=True,
    static_reconst_iteration=1500,
)

ModelParams = dict(
 sh_degree=3
)
