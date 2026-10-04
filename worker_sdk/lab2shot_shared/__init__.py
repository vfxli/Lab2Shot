"""Lab2Shot shared library: a single implementation used by both the core and the workers. Depends only on NumPy and
the standard library, and never imports lab2shot or lab2shot_worker. Installed into every extension environment with
the worker SDK (worker_sdk/pyproject.toml) and used directly by the core.

    protocol      job schema, event prefix, message codes, Failure, shown
    poses         camera rotations, similarity fitting, pose blending and interpolation, unprojection
    smoothing     zero-phase temporal smoothing of values, angles and quaternions
    motion        skeletal motion: quaternions, hierarchies, retargeting, resampling, keys, foot contacts
    smpl          SMPL-family skeletons and conversion between parameters and skeletal animation
    rig_motion    files of the rig-and-model contract
    light_probe   light-probe conventions
    exr           EXR writer
    gaussians     3D gaussian covariance, real SH basis and the bake of a transform into a splat set
    scene_arrays  3D data exchanged between the core and format workers as plain arrays
    names         names in 3D files <-> identifiers (USD, Alembic) and the sibling tie-break (every writer)
    body_models   locations of manually downloaded body models
    gpu_arch      GPU architectures supported by an environment's torch and kernels
    memory        available system memory
"""
