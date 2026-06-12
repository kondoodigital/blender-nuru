# Nuru: Contains Nuru-specific changes relative to the Blender parent.

import bpy

eevee = bpy.context.scene.eevee
options = eevee.ray_tracing_options

eevee.use_raytracing = True
eevee.ray_tracing_method = 'HARDWARE'
eevee.use_hardware_raytracing_gi = True
options.resolution_scale = 'PERCENT_50'
options.gi_spatial_samples = '16'
eevee.hardware_raytracing_reflection_mode = 'FULL'
eevee.ray_tracing_reflection_bounces = 3
eevee.hardware_raytracing_refraction_mode = 'FULL'
eevee.ray_tracing_refraction_bounces = 3
eevee.use_hardware_raytracing_shadows = True
eevee.hardware_raytracing_shadow_samples = 2
eevee.hardware_raytracing_shadow_transparency = 0.5
eevee.hardware_raytracing_shadow_color_intensity = 0.5
eevee.indirect_light_intensity = 5.0
eevee.clamp_surface_indirect = 10.0
options.use_denoise = True
options.denoise_filter = 'OIDN'
options.denoise_input_passes = 'RGB_ALBEDO_NORMAL'
options.denoise_prefilter = 'FAST'
options.denoise_quality = 'BALANCED'
options.denoise_sample_interval = '1'
options.denoise_use_gpu = True
options.denoise_spatial = True
options.denoise_temporal = True
