
OptimizationParams = dict(
    densification_interval=200,
    densify_from_iter=1000,
    densify_until_iter=20_000,
    #
    densify_grad_threshold_fine_init=0.001,
    densify_grad_threshold_after=0.005,
    #
    pruning_from_iter=1000,
    pruning_interval=200,
    opacity_threshold_fine_init=0.001,
    opacity_threshold_fine_after=0.005,

    initial_gaussians=1000,
    no_coarse=True,
)
