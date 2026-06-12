# Nuru Shadows And Lights

Nuru can use hardware ray tracing for light shadows. This helps shadows respond to scene geometry and transparent materials in the **Rendered** viewport and final render.

## Raytrace Shadows

Enable **Raytrace Shadows** in **Render Properties → Eevee → Raytracing → Method: Nuru Raytracing → Quick Settings**.

When **Raytrace Shadows** is on, Nuru uses hardware ray tracing for shadow visibility. When it is off, Blender uses the available standard Eevee shadow behavior.

## Samples

**Samples** controls soft shadow quality.

| Lower values | Higher values |
| --- | --- |
| Faster, more likely to show noise. | Smoother, more expensive. |

Start with a low value while working, then raise it when checking the final look.

## Transparent Shadows

**Transparent Shadows** controls how much transparent material affects shadow strength.

| Value | Result |
| --- | --- |
| 0 | Transparent materials behave more like opaque blockers in shadows. |
| 0.5 | Transparency partly affects the shadow. |
| 1 | Transparency is fully considered. |

This is useful for glass, alpha surfaces, and tinted transparent materials.

## Color Transmission

**Color Transmission** controls how strongly transparent materials tint their shadows.

| Value | Result |
| --- | --- |
| 0 | Shadow tint is reduced. |
| 1 | The material color is preserved more strongly in the shadow. |

Use this when you want colored glass or transparent surfaces to cast colored light into the shadow.

Thin Glass materials are exempt from ray-traced shadows entirely: window panes pass light at full strength instead of casting a shadow.

## Local Lights, Sun, And World Light

Nuru is designed to work with local lights, sun lights, and world lighting. The visible result still depends on the light type, material setup, scene scale, and the enabled Quick Settings.

For best results, check shadows in **Rendered** viewport with the same feature settings you intend to render with.

## Volumes

Volumetrics render in Nuru. Lit fog responds to lights and shadows, including sun shafts through windows and blinds.

## Related Pages

- [UI Reference](nuru_ui_reference.html)
- [Nuru Overview](nuru_overview.html)
- [Materials](nuru_materials.html)
