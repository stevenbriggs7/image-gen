# Generator ideas backlog

## Done
- Lines (Perlin flow field, short strokes)
- Circles (noise-modulated radii)
- Wave (sine wave spine, perpendicular lines)
- Pendulum (damped precessing ellipse / rosette)
- Shapes (circles, triangles, squares mixed)
- Attractor (strange attractor — in progress)

## Candidate ideas

### Flow Streamlines
Trace long continuous curves following the existing Perlin noise field (like ink dropped in water). Complement to the Lines generator — same maths, but curves that travel across the whole canvas and build up layered, river-like structures. Reuses `_fractal_noise_field` and the flow-curve step loop already in `generate.py`.

### Voronoi Cells
Partition the canvas into cells using scattered seed points and draw their outlines or fills. Creates mosaic, stained-glass, cracked-earth or biological cell patterns. Noise can modulate cell size and density for organic clustering. Could use SciPy's `Voronoi` or a hand-rolled nearest-neighbour approach.

### Spirograph
Parametric curves from a point on a circle rolling inside or outside another circle (hypotrochoid / epitrochoid). Like the classic toy — produces precise geometric flower and star patterns. Natural companion to the Pendulum generator. Pure maths, very fast to compute.
