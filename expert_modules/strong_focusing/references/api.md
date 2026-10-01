# Python API and execution

In a configured Brilliance research program:

```python
import brilliance.accelerator_api as api
expert = api.expert_module("laboratory.strong_focusing")

# An illustrative 90-degree map, not a tracked lattice result.
report = expert.periodic_plane_report(
    [[0.0, 2.0], [-0.5, 0.0]], symplectic_tolerance=1e-10
)
assert report["periodic_solution_defined"]
assert report["beta_m"] == 2.0
```

`periodic_plane_report(matrix, *, symplectic_tolerance)` accepts a finite 2-by-2
map and a positive absolute tolerance on `abs(det(M)-1)`. It returns determinant,
symplectic error, half-trace, `trace_stability_margin = 1-abs(trace/2)`, and
whether a strictly elliptic periodic solution is defined. When defined it also
returns `beta_m`, `alpha`, `gamma_per_m`, wrapped phase in radians and its fraction
of 2*pi. Otherwise those fields are `None`. A smaller positive trace margin means
closer to a linear stability boundary; it is not a nonlinear acceptance metric.

Use the configured backend for actual dynamics:

```python
jt = api.jutrack()
# lattice is the project's existing pyJuTrack lattice, or:
# lattice = api.build_jutrack_lattice(candidate)
matrix6 = jt.findm66(lattice, dp=0.0)
twiss = jt.twissring(lattice, dp=0.0)
```

These backend calls are source-checked, not executed in this package's build
environment. Set species/energy and reference orbit using the configured backend
contract. For an error lattice, linearize at the relevant closed orbit. Before
passing `matrix6[:2, :2]` or `matrix6[2:4, 2:4]` to the helper, verify negligible
coupling and fixed reference momentum. Use coupled analysis when needed.
`twissring` is a periodic solution; use `twissline` with explicit entrance Twiss
for a nonperiodic channel. A periodic cell can be analyzed without claiming it
is a physically closed ring.

Existing `api.measure_response` and `api.bounded_response_correction` can support
matching callbacks. Every trial must reconstruct/update a private candidate,
preserve bounds and geometry, then remeasure. No optimizer or acceptance gate is
embedded in this module. Check numerical tolerances through map/integration
convergence and choose physical tolerances from the project requirements.
