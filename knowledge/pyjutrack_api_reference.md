# pyJuTrack API Reference

pyJuTrack is the Python wrapper for the JuTrack.jl accelerator modeling
library. All lattice elements, analysis functions, and tracking utilities
are accessed through a single `jt` module object.

## Setup

```python
import pyJuTrack as jt
```

---

## 1. Element Constructors

### Critical Naming Convention
- Python constructor parameter: `length` (positional)
- Julia field on the created object: `.len`
- NEVER use `.L` or `.length` to read back the length — always use `.len`
- The bending angle field is `.angle`, NOT `.rad`
- The `.rad` field is a radiation flag (0 or 1), not an angle

### MARKER
```python
m = jt.MARKER(name)
# m.name, m.len (always 0)
```

### DRIFT
```python
d = jt.DRIFT(name, length)
# d.name, d.len
# Optional: T1, T2, R1, R2, RApertures, EApertures
```

### KQUAD (Canonical Quadrupole — kick-drift-kick symplectic integrator)
```python
q = jt.KQUAD(name, length, k1)
# q.name, q.len, q.k1
# Optional: k0=0, k2=0, k3=0, PolynomA, MaxOrder=1, NumIntSteps=10,
#   rad=0, FringeQuadEntrance=0, FringeQuadExit=0,
#   T1, T2, R1, R2, RApertures, EApertures, KickAngle
```
This is the standard quadrupole element recommended for tracking and
optimization. Uses symplectic integration.

### QUAD (Matrix-formalism Quadrupole)
```python
q = jt.QUAD(name, length, k1)
# q.name, q.len, q.k1
# Optional: rad=0, T1, T2, R1, R2, RApertures, EApertures
```
Uses linear matrix transport. Faster but less accurate for nonlinear studies.

### KSEXT (Canonical Sextupole)
```python
s = jt.KSEXT(name, length, k2)
# s.name, s.len, s.k2
# Default MaxOrder=2. Same optional kwargs as KQUAD.
```

### KOCT (Canonical Octupole)
```python
o = jt.KOCT(name, length, k3)
# o.name, o.len, o.k3
# Default MaxOrder=3. Same optional kwargs as KQUAD.
```

### SBEND (Sector Bend)
```python
b = jt.SBEND(name, length, angle)
# b.name, b.len, b.angle, b.e1, b.e2
# Optional: e1=0, e2=0, PolynomA, PolynomB (multipole components),
#   MaxOrder (auto-set from PolynomB), NumIntSteps=10, rad=0,
#   fint1=0, fint2=0, gap=0, FringeBendEntrance=1, FringeBendExit=1,
#   FringeQuadEntrance, FringeQuadExit, T1, T2, R1, R2,
#   RApertures, EApertures, KickAngle
```
IMPORTANT: `angle` is the bending angle in RADIANS.
For combined-function dipoles, add quadrupole gradient via
`PolynomB=[0, k1, 0, 0]` (index 0=dipole kick, 1=quad, 2=sext, 3=oct).

### RBEND (Rectangular Bend)
```python
b = jt.RBEND(name, length, angle)
```
Automatically sets `e1 = angle/2`, `e2 = angle/2`, then creates an SBEND
internally. Same fields as SBEND. Use RBEND when the magnet has flat
entrance and exit faces perpendicular to the design orbit.

### ESBEND (Exact Sector Bend)
```python
b = jt.ESBEND(name, length, angle)
# Extra field: .gK (geometric K). Optional: gK=0, PolynomB, etc.
```
Uses exact curved-geometry Hamiltonian. More accurate for large bend angles.

### ERBEND (Exact Rectangular Bend)
```python
b = jt.ERBEND(name, length, angle)
```
Sets `e1 = angle/2`, `e2 = angle/2`, creates ESBEND internally.

### RFCA (RF Cavity)
```python
rf = jt.RFCA(name, length, volt, freq)
# rf.volt (Volts), rf.freq (Hz), rf.h (harmonic number)
# Optional: h=1.0, lag=0.0, philag=0.0, energy=0.0
```

### CORRECTOR / HKICKER / VKICKER
```python
c = jt.CORRECTOR(name, length=0.0, xkick=0, ykick=0)
h = jt.HKICKER(name, length=0.0, xkick=0)
v = jt.VKICKER(name, length=0.0, ykick=0)
```

### SOLENOID
```python
sol = jt.SOLENOID(name, length, ks)
# ks: solenoid strength in rad/m
```

### thinMULTIPOLE
```python
m = jt.thinMULTIPOLE(name, PolynomB=[0, 0, k2, 0])
# PolynomB indices: [0]=dipole, [1]=quad, [2]=sext, [3]=oct
# MaxOrder auto-detected from highest nonzero PolynomB entry
```

### LongitudinalRLCWake
```python
w = jt.LongitudinalRLCWake(name="RLCWake", freq=..., Rshunt=..., Q0=...)
```

---

## 2. Lattice Class

```python
lat = jt.Lattice(elements_list)    # Create from list of elements
lat = jt.Lattice()                 # Empty lattice

# Operations
len(lat)                # Number of elements
lat[i]                  # Access element (0-based Python index)
lat[i:j]               # Slicing
lat * N                 # Repeat lattice N times (full ring from one cell)
N * lat                 # Same
lat1 + lat2            # Concatenate two lattices
lat.append(elem)       # Add one element
lat.extend(elems)      # Add multiple elements or another Lattice
lat.insert(i, elem)    # Insert at position
lat.copy()             # Shallow copy
lat.deepcopy()         # Deep copy (independent elements)
lat.total_length()     # Total length in meters
lat.spos(indices=None) # s-positions array (0-based indices)

# Iteration
for elem in lat:
    print(elem.name, elem.len)
```

### Building a Ring from a Cell

The standard pattern for constructing a full ring:
```python
# Define one cell
cell = jt.Lattice([d1, qf, d2, bend, d3, qd, d4, bend, d3, qf, d2, ...])

# Build full ring
N_cells = 20
ring = cell * N_cells

# Verify total length
print(f"Circumference: {ring.total_length():.3f} m")
```

### Important: Bending Angle Consistency

When building a cell for a ring, the total bending angle of ALL dipoles in
one cell MUST equal exactly 2π / N_cells. For example:
- 20 cells → each cell bends 2π/20 = 0.31416 rad total
- If a cell has 7 dipoles sharing this angle, each dipole angle = 0.31416/7
- If using reverse bends, the main dipole angles must compensate:
  Σ(main angles) + Σ(reverse angles) = 2π / N_cells

---

## 3. Beam Class

```python
import numpy as np

# From explicit particle coordinates
r = np.zeros((n_particles, 6))   # [x, px, y, py, z, δp/p0]
beam = jt.Beam(r, energy=3e9)

# Gaussian distribution
beam = jt.Beam_Gauss(
    nmacro=1000, energy=3e9,
    betax=10.0, alphax=0.0, emitx=1e-9,
    betay=5.0,  alphay=0.0, emity=1e-11,
    betaz=1.0,  alphaz=0.0, emitz=1e-6
)

# Properties
beam.r           # numpy array (n × 6)
beam.energy      # beam energy in eV
beam.nmacro      # number of macro-particles
beam.lost_flag   # boolean array— True for lost particles
```

---

## 4. Tracking

```python
# Single-pass tracking through a lattice
jt.linepass(lattice, beam)

# Track and save coordinates at specific elements (0-based indices)
coords = jt.linepass(lattice, beam, refpts=[0, 10, 20])
# coords is a list of numpy arrays

# Multi-turn tracking
jt.ringpass(lattice, beam, num_turns=1000)

# Parallel tracking (auto-detects threads if nthreads=None)
jt.plinepass(lattice, beam)
jt.pringpass(lattice, beam, num_turns=1000)

# Check which particles survived
lost = jt.check_lost(beam)   # boolean array
```

---

## 5. Optics (Twiss Parameters)

### Ring Twiss (periodic solution)
```python
twiss_list = jt.twissring(lattice, dp=0.0, refpts=None)
```
Returns a list of EdwardsTengTwiss objects at each element exit.

**IMPORTANT**: `refpts` uses **1-based Julia indexing** when calling twissring.
If `refpts=None`, returns Twiss at ALL element exits.

Access Twiss fields:
```python
tw = twiss_list[0]
tw.betax, tw.betay           # beta functions [m]
tw.alphax, tw.alphay         # alpha functions
tw.dx, tw.dpx, tw.dy, tw.dpy  # dispersion [m] and its derivative
tw.mux, tw.muy               # phase advance [rad] up to this point
tw.s                          # s-position [m]
```

### Periodic Twiss at Entrance
```python
tw0 = jt.periodicEdwardsTengTwiss(lattice, dp=0.0, order=0)
# Returns the Twiss at the entrance with periodic boundary conditions
```

### Propagate Twiss Through a Line
```python
tw_in = jt.EdwardsTengTwiss(betax=10.0, betay=5.0, alphax=0.0, alphay=0.0)
twiss_list = jt.twissline(tw_in, lattice, dp=0.0, order=0, refpts=None)
```

---

## 6. Transfer Matrix

```python
m66 = jt.findm66(lattice, dp=0.0, order=0)
# Returns 6×6 numpy array (one-turn transfer matrix)

m66_fast = jt.fastfindm66(lattice, dp=0.0)
# Faster version using matrix multiplication only

# Transfer matrices at specific locations
matrices = jt.findm66_refpts(lattice, refpts, dp=0.0, order=0)
```

---

## 7. Closed Orbit

```python
x_co, M = jt.find_closed_orbit(lattice,
    dp=0.0, mass=jt.m_e, energy=3e9,
    guess=np.zeros(6), max_iter=20, tol=1e-8)
# x_co: 6-element closed orbit vector
# M: 6×6 one-turn matrix at the closed orbit

# 4D closed orbit (transverse only, fixed dp)
x_co, M = jt.find_closed_orbit_4d(lattice, dp=0.0)

# 6D closed orbit (includes longitudinal)
x_co, M = jt.find_closed_orbit_6d(lattice)
```

---

## 8. Tune and Chromaticity

```python
nux, nuy = jt.gettune(lattice, dp=0.0)
# Returns fractional tunes (values between 0 and 0.5)

xi_x, xi_y = jt.getchrom(lattice, dp=0.0, energy=3e9)
# Returns chromaticity (Δν/Δδ)
```

---

## 9. Ring Parameters (Comprehensive)

```python
params = jt.ringpara(lattice, energy=3e9)
# With RF cavity:
params = jt.ringpara(lattice, energy=3e9, Vrf=3e6, harm=400, freq_rf=500e6)
```

Returns a dict with ALL ring parameters:
```python
params["E0_GeV"]          # energy in GeV
params["circumference"]   # circumference [m]
params["nux"], params["nuy"]  # tunes
params["chromx"], params["chromy"]  # chromaticity
params["alphac"]          # momentum compaction factor
params["emittx"]          # natural emittance [m·rad]
params["U0"]              # energy loss per turn [eV]
params["Jx"], params["Jy"], params["Je"]  # damping partition numbers
params["I1"]–params["I5"] # radiation integrals
params["sigma_E"]         # natural energy spread
params["dampingtime_x"]   # horizontal damping time [s]
params["dampingtime_y"]   # vertical damping time [s]
params["dampingtime_E"]   # energy damping time [s]
# If Vrf > 0:
params["phi_s"]           # synchronous phase [rad]
params["nus"]             # synchrotron tune
params["delta_max"]       # RF bucket height
params["bunchlength"]     # natural bunch length [m]
```

This is the most important analysis function for evaluating a lattice design.
Always call ringpara after building a ring and verifying stability.

---

## 10. Radiation Control

```python
jt.rad_on()    # Enable radiation effects in tracking
jt.rad_off()   # Disable radiation effects

# Energy loss calculation
u0 = jt.tracking_U0(lattice, energy=3e9, mass=jt.m_e)
u0 = jt.integral_U0(lattice, energy=3e9, mass=jt.m_e)
```

---

## 11. TPSA (Truncated Power Series Algebra) for Optimization

TPSA enables automatic differentiation through the lattice, critical for
gradient-based optimization of magnet strengths.

```python
# Set up TPSA
jt.set_tps_dim(n)              # n = number of optimization variables

# Create TPSA variables
k1_tpsa = jt.DTPSAD(k1_value, var_idx)  # var_idx is 1-based

# Convert lattice to TPSA type for AD tracking
lat_tpsa = jt.Number2TPSAD(jt.Lattice(cell))

# Set TPSA field on element
jt.set_field(lat_tpsa[i], 'k1', k1_tpsa)

# Compute gradient
grad = jt.Gradient(objective_function, x_array)     # → numpy gradient
jac = jt.Jacobian(vector_function, x_array)          # → numpy Jacobian

# Convert back to numeric lattice
lat_numeric = jt.TPSAD2Number(lat_tpsa)
```

### Optimization Pattern (Tune Matching Example)

```python
import numpy as np

cell = jt.Lattice([...])  # define cell with initial k1 values
ring = cell * N_cells

# Variables: k1 values of quadrupole families
k1_init = np.array([qf.k1, qd.k1])
target_tunes = np.array([nux_target, nuy_target])

def tune_residual(k1_vec):
    jt.set_tps_dim(len(k1_vec))
    k1_tpsa = [jt.DTPSAD(k1_vec[i], i+1) for i in range(len(k1_vec))]
    cell_tpsa = jt.Number2TPSAD(cell.deepcopy())
    # Set k1 on each quadrupole family
    for idx in qf_indices:
        jt.set_field(cell_tpsa[idx], 'k1', k1_tpsa[0])
    for idx in qd_indices:
        jt.set_field(cell_tpsa[idx], 'k1', k1_tpsa[1])
    ring_tpsa = cell_tpsa * N_cells
    # Compute Twiss
    twiss = jt.twissring(ring_tpsa)
    nux = twiss[-1].mux / (2 * np.pi)
    nuy = twiss[-1].muy / (2 * np.pi)
    return np.array([(nux - target_tunes[0])**2 + (nuy - target_tunes[1])**2])

# Newton iteration
k1 = k1_init.copy()
for _ in range(50):
    grad = jt.Gradient(tune_residual, k1)
    val = tune_residual(k1)
    if np.max(np.abs(val)) < 1e-12:
        break
    k1 -= 0.5 * grad * val[0]
```

---

## 12. Resonance Driving Terms

```python
dlist, tune = jt.computeRDT(lattice,
    indices=[...],         # element indices (1-based) for sextupoles
    E0=3e9,
    chromatic=True,        # include chromatic RDTs
    coupling=True,         # include coupling RDTs
    geometric1=True,       # include first-order geometric RDTs
    geometric2=True,       # include second-order geometric RDTs
    tuneshifts=True        # include amplitude-dependent tune shifts
)
# dlist[i].h21000[0] — RDT h21000
# dlist[i].h11001[0] — chromatic term
# tune: (nux, nuy)
```

---

## 13. Dynamic Aperture

```python
DA, survived = jt.dynamic_aperture(lattice,
    nturns=1000,      # number of turns to track
    amp_max=0.01,     # max amplitude [m]
    amp_step=0.0005,  # step size [m]
    angle_steps=20,   # number of angles in (x,y) plane
    E=3e9,            # energy [eV]
    dp=0.0            # momentum deviation
)
# DA: array (n_angles × 2) with boundary [x, y] in meters
# survived: all surviving particles
```

---

## 14. Beam Analysis Utilities

```python
tw = jt.twiss_beam(beam)
# tw = {"betax": ..., "alphax": ..., "emitx": ...,
#        "betay": ..., "alphay": ..., "emity": ...,
#        "betaz": ..., "alphaz": ..., "emitz": ...}

ex, ey, ez = jt.get_emittance(beam)
centroid = jt.get_centroid(beam)       # [6] array
moment2nd = jt.get_2nd_moment(beam)    # [6×6] array
```

---

## 15. Lattice Utilities

```python
L = jt.total_length(lattice)
s = jt.spos(lattice)  # s-position array

# Find elements by type
quad_indices = jt.findelem(lattice, element_type=jt.element_types.KQUAD)

# Find elements by name
qf_indices = jt.findelem(lattice, name="QF")

# Find elements by field value
indices = jt.findelem(lattice, field="k1", value=1.2)

# NOTE: findelem returns 1-based Julia indices

# Save/load lattice
jt.save_lattice(lattice, "file.jls")
lattice = jt.load_lattice("file.jls")

# Copy elements
elem_copy = jt.copy_element(elem)
elem_deep = jt.deepcopy_element(elem)
```

---

## 16. Physical Constants

```python
jt.m_e              # Electron mass: 0.51099895069e6 eV
jt.m_p              # Proton mass: 938.27208816e6 eV
jt.charge_e         # Elementary charge: 1.602176634e-19 C
jt.speed_of_light   # Speed of light: 2.99792458e8 m/s
jt.CGAMMA           # Radiation constant: 8.846e-5 m/eV³
jt.Cq               # Quantum constant: 3.8319e-13 m
```

---

## 17. Common Patterns and Recipes

### Build a FODO Cell
```python
angle_per_cell = 2 * np.pi / N_cells
angle_per_dipole = angle_per_cell / 2
L_bend = 2.0  # dipole length in meters

qf = jt.KQUAD("QF", 0.3, k1=1.5)
qd = jt.KQUAD("QD", 0.3, k1=-1.5)
d1 = jt.DRIFT("D1", 0.5)
b = jt.SBEND("B", L_bend, angle_per_dipole)
half_qf = jt.KQUAD("HQF", 0.15, k1=1.5)

cell = jt.Lattice([half_qf, d1, b, d1, qd, d1, b, d1, half_qf])
ring = cell * N_cells
```

### Build an MBA Cell (e.g. 7BA)
```python
N_cells = 20
angle_per_cell = 2 * np.pi / N_cells
n_bends = 7
angle_per_dipole = angle_per_cell / n_bends

# Main dipoles
b = jt.SBEND("B", 1.5, angle_per_dipole)
b_match = jt.SBEND("BM", 0.8, angle_per_dipole)  # shorter matching bends

# Quadrupoles
qf = jt.KQUAD("QF", 0.25, k1=2.0)
qd = jt.KQUAD("QD", 0.25, k1=-2.0)
qm1 = jt.KQUAD("QM1", 0.25, k1=3.0)

# Sextupoles
sf = jt.KSEXT("SF", 0.15, k2=50.0)
sd = jt.KSEXT("SD", 0.15, k2=-50.0)

# Drifts
d_long = jt.DRIFT("DL", 2.5)   # ID straight half-length
d_short = jt.DRIFT("DS", 0.15)
d_mid = jt.DRIFT("DM", 0.3)

# Build half-cell (mirror symmetric)
# matching section → central arc → mirror matching
half_cell = [d_long, qm1, d_short, sf, d_short, b_match, d_mid,
             qd, d_short, sd, d_short, b, d_mid,
             qf, d_short, b, d_mid]

cell = jt.Lattice(half_cell + half_cell[::-1])
ring = cell * N_cells
```

### Evaluate a Ring
```python
# 1. Check stability
try:
    twiss = jt.twissring(ring)
    print("Ring is stable")
except:
    print("Ring is UNSTABLE — Twiss computation failed")
    # Adjust quadrupole strengths

# 2. Get tunes
nux, nuy = jt.gettune(ring)
print(f"Tunes: ({nux:.4f}, {nuy:.4f})")

# 3. Get chromaticity
xi_x, xi_y = jt.getchrom(ring, energy=3e9)
print(f"Chromaticity: ({xi_x:.1f}, {xi_y:.1f})")

# 4. Full ring parameters
params = jt.ringpara(ring, energy=3e9)
print(f"Emittance: {params['emittx']*1e12:.1f} pm·rad")
print(f"Momentum compaction: {params['alphac']:.2e}")
print(f"Energy loss/turn: {params['U0']/1e6:.3f} MeV")

# 5. Plot Twiss
import matplotlib.pyplot as plt
s = [tw.s for tw in twiss]
betax = [tw.betax for tw in twiss]
betay = [tw.betay for tw in twiss]
dx = [tw.dx for tw in twiss]

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
ax1.plot(s, betax, label=r'$\beta_x$')
ax1.plot(s, betay, label=r'$\beta_y$')
ax1.set_ylabel('Beta [m]')
ax1.legend()
ax2.plot(s, dx, label=r'$\eta_x$', color='green')
ax2.set_ylabel('Dispersion [m]')
ax2.set_xlabel('s [m]')
ax2.legend()
plt.tight_layout()
plt.savefig("twiss.png", dpi=150)
```

### Tune Matching with Newton Method
```python
def match_tunes(cell, N_cells, qf_indices, qd_indices, target_nux, target_nuy):
    """Match tunes using TPSA-based Newton iteration."""
    ring = cell * N_cells
    k1_qf = cell[qf_indices[0]].k1
    k1_qd = cell[qd_indices[0]].k1
    k1 = np.array([k1_qf, k1_qd], dtype=float)

    for iteration in range(100):
        jt.set_tps_dim(2)
        cell_copy = cell.deepcopy()
        cell_tpsa = jt.Number2TPSAD(cell_copy)
        k1_ad = [jt.DTPSAD(k1[i], i+1) for i in range(2)]
        for idx in qf_indices:
            jt.set_field(cell_tpsa[idx], 'k1', k1_ad[0])
        for idx in qd_indices:
            jt.set_field(cell_tpsa[idx], 'k1', k1_ad[1])
        ring_tpsa = cell_tpsa * N_cells

        try:
            twiss = jt.twissring(ring_tpsa)
        except:
            k1 *= 0.95  # reduce if unstable
            continue

        nux = twiss[-1].mux / (2 * np.pi)
        nuy = twiss[-1].muy / (2 * np.pi)
        residual = np.array([nux - target_nux, nuy - target_nuy])

        if np.max(np.abs(residual)) < 1e-6:
            break

        jac = jt.Jacobian(lambda x: ..., k1)  # or compute manually
        dk = np.linalg.solve(jac, -residual)
        k1 += 0.5 * dk  # damped step

    # Apply final values
    for idx in qf_indices:
        cell[idx].k1 = float(k1[0])
    for idx in qd_indices:
        cell[idx].k1 = float(k1[1])
    return cell
```

---

## 18. Index Conventions Summary

| Function / Object | Index Convention |
|---|---|
| `lat[i]` | 0-based Python |
| `linepass refpts` | 0-based Python (auto-converted) |
| `twissring refpts` | 1-based Julia (NOT auto-converted) |
| `findelem` returns | 1-based Julia |
| `PolynomB` / `PolynomA` arrays | 1-based: [dipole, quad, sext, oct] |
| `DTPSAD(val, var_idx)` | 1-based variable index |

---

## 19. Plotting

```python
fig, ax, handles = jt.plot_lattice(lattice,
    width=0.25,
    axis=True,
    savepath="lattice_layout.png",
    figsize=(8, 8),
    layout="curved",    # or "straight"
    show=False
)
```
