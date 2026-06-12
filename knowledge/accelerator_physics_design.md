# Accelerator Physics Design Knowledge

## Emittance Theory and Scaling

The natural emittance of an electron storage ring is governed by a balance
between quantum excitation and radiation damping. The theoretical minimum
emittance (TME) of a single bending cell is:

    ε_TME = C_q · γ² · θ³ / (12√15 · J_x)

where C_q ≈ 3.832×10⁻¹³ m is the quantum constant, γ is the Lorentz factor,
θ is the bending angle per dipole, and J_x is the horizontal damping partition
number (typically ≈ 1 for an isomagnetic ring).

A convenient numerical form:

    ε_TME [pm·rad] ≈ 8.24×10⁻³ · γ² · (θ [rad])³ / J_x

Achieving the TME requires specific Twiss parameters at the dipole center:
α = 0, η' = 0, and optimal β and η values that depend on the dipole length.

For an N-bend achromat (NBA) with equal dipoles sharing the total bending
2π among N_cells cells (each with N bends), the bare-lattice emittance scales
as:

    ε ∝ γ² / (J_x · N_d³)

where N_d = N × N_cells is the total number of dipoles. Doubling N_d reduces
emittance eightfold — this is the fundamental motivation for multi-bend
achromat (MBA) lattices.

The fraction of TME achieved depends on the optics at the dipole:
- In a conventional DBA cell: ε/ε_TME ≈ 3 (dispersion is zeroed, not optimized)
- quasi-TME cell (non-zero dispersion at straights): ε/ε_TME ≈ 1.2–1.5
- TME cell with perfect match: ε/ε_TME = 1.0

Modern rings target tens of pm·rad or below. The "diffraction limit" for
X-rays at λ = 1 Å is ε ≈ λ/(4π) ≈ 8 pm·rad.


## Multi-Bend Achromat Lattice Types

### Conventional MBA

A conventional N-bend achromat (e.g. 5BA, 7BA, 9BA) arranges N identical
dipoles separated by matching quadrupoles and sextupoles. The outer dipoles
(at cell ends) are "matched" so that dispersion is zeroed at the insertion
device straight sections, making the cell achromatic.

Typical MBA structure:
- Matching section (dispersion suppressor at cell ends)
- Central arc of quasi-TME cells
- ID straight sections at both ends with η_x = 0

Cell count trade-off: more bends → lower emittance but shorter straights,
stronger chromatic effects, and more difficult nonlinear optimization.

### Hybrid MBA (HMBA)

The hybrid multi-bend achromat is the dominant modern approach pioneered for
ESRF-EBS. It combines:
- A central arc of unit cells close to the TME condition (with longitudinal
  gradient bends and/or reverse bends)
- Dedicated chromatic correction sections at cell ends using sextupole pairs
  in -I geometry
- Matching quadrupole triplets between the arc and the ID straights

Key advantages of HMBA over conventional MBA:
1. Sextupoles are concentrated in the dispersion bumps near cell edges, away
   from the low-dispersion arc. This separates chromatic correction from the
   emittance-minimizing arc.
2. The -I transform between sextupole pairs cancels geometric aberrations to
   first order.
3. Dispersion leaks from the arc boost η at the sextupole locations, reducing
   the required sextupole strengths.

Facilities using HMBA concepts:
- ESRF-EBS (H7BA, 6 GeV, 32 cells, ~130 pm·rad)
- APS-U (H7BA, 6 GeV, 40 cells, ~42 pm·rad)
- HEPS (H7BA, 6 GeV, 48 cells, ~34 pm·rad)
- SLS-2 (H7BA, 2.4 GeV, 12 cells, ~100 pm·rad)
- Diamond-II (H6BA, 3.5 GeV, 24 cells, ~160 pm·rad)
- Sirius (5BA with chromatic sextupoles, 3 GeV, 20 cells, ~250 pm·rad)
- MAX IV (7BA, 3 GeV, 20 cells, ~330 pm·rad)


## Techniques for Reducing Emittance

### Combined-Function Dipoles

Adding a quadrupole gradient to a bending dipole (combined-function dipole)
provides extra focusing that shifts the optimal β and η at the dipole and
allows achieving lower emittance with fewer independent quadrupoles. However,
the gradient must be carefully balanced to avoid exciting chromaticity.

### Longitudinal Gradient Bends (LGB)

A longitudinal gradient bend has its bending field vary along its length —
stronger at the center (where low dispersion means less quantum excitation)
and weaker at the edges. In practice this can be approximated by splitting one
long dipole into slices with different fields, or by using a continuously
varying pole-gap magnet. LGBs can reduce emittance below the uniform-field
TME limit by ~30–40%.

### Reverse Bends (RB)

A reverse bend is a short, negative-angle dipole placed between the main
bends. Its negative dispersion contribution reduces the average dispersion in
the arc, lowering emittance. The total bending angle of the cell increases
because the main bends must compensate for the negative angle:

    Σ θ_main + Σ θ_RB = 2π / N_cells

Since the RBs add path length but negative bending, ε can drop significantly
(30–50% further) at the cost of a slightly longer circumference and higher
energy loss per turn.

### Damping Wigglers

Damping wigglers in long straight sections increase the synchrotron radiation
energy loss, enhancing radiation damping and further reducing the equilibrium
emittance. Effective when the ring has enough straight-section length.

### Adjusting the Damping Partition Number J_x

Robinson's theorem constrains J_x + J_y + J_z = 4 (for an isomagnetic ring
without RF). By shifting the damping partition (e.g., via off-center orbit in
combined-function dipoles or via gradient dipoles), one can increase J_x
beyond 1, thereby lowering horizontal emittance at the expense of longitudinal
damping.


## Nonlinear Dynamics and Dynamic Aperture

### The Chromaticity Problem

Every quadrupole contributes positive natural chromaticity:

    ξ_nat ∝ -∮ β · K_quad ds / (4π)

MBA cells with very strong focusing produce large negative natural
chromaticity (ξ ~ -100 or more), requiring strong sextupoles to correct ξ
to small positive values (~+2). These sextupoles drive amplitude-dependent
tune shifts (ADTS) and resonance excitation that limit the dynamic aperture.

### The -I Transformation

The fundamental sextupole-cancellation scheme: placing two identical
sextupoles separated by a betatron phase advance of exactly π (i.e., a -I
transfer matrix between them) cancels all geometric kicks to first order.
The sextupole pair acts as a pure chromatic corrector.

Requirements:
- Phase advance Δμ_x = π, Δμ_y = π between the sextupoles
- Identical β_x, β_y, η at both sextupole locations
- In practice, small deviations from -I degrade cancellation → careful
  matching of the transport between sextupole pairs is essential

### Higher-Order Achromat (HOA) Cancellation

The HOA method cancels resonance driving terms (RDTs) by choosing the
betatron phase advance per cell so that nonlinear kicks from repeated cells
add up to zero over one superperiod. For a cell with 5 sextupoles, certain
symmetries (e.g., making the non-interleaved sextupole pairs satisfy -I over
the superperiod) cancel all first- and some second-order geometric RDTs.

### Resonance Driving Terms (RDTs)

Sextupoles excite the following first-order geometric RDTs:
- h₂₁₀₀₀ (third-integer horizontal, 3ν_x)
- h₃₀₀₀₀ (third-integer horizontal, 3ν_x)
- h₁₀₁₁₀ (sum coupling, ν_x + 2ν_y)
- h₁₀₂₀₀ (difference coupling, ν_x - 2ν_y)

And second-order (octupole-like) amplitude-dependent tune shifts:
- Δν_x ∝ J_x (horizontal ADTS)
- Δν_y ∝ J_y (vertical ADTS)
- Cross-term Δν_x ∝ J_y and Δν_y ∝ J_x

Minimizing or cancelling these terms is the primary strategy for enlarging
dynamic aperture.

### Working Point Selection

The working point (ν_x, ν_y) must avoid low-order resonances m·ν_x + n·ν_y = p
where |m| + |n| ≤ 4 (or even 5 for ultra-low-emittance rings). Common choices
place the tunes between the half-integer and third-integer resonances. The
fractional tunes should not be near 1/3, 1/4, or 1/5.

For MBA lattices with N_cells identical cells, the integer part of the tune
is approximately:

    ν_int ≈ N_cells × μ_cell / (2π)

where μ_cell is the phase advance per cell.


## Design Workflow for a 4th-Generation Light Source

### Stage 1: Linear Lattice Design

1. Choose energy based on photon requirements (soft X-ray: 2–3 GeV,
   hard X-ray: 3–6 GeV, or higher for very high brightness).
2. Decide cell type (e.g., H7BA, 5BA, 9BA) and number of cells based on
   circumference constraints and emittance target.
3. Design one cell:
   a. Set bending angles so that angles sum to 2π/N_cells.
   b. Design the central arc to approach TME conditions.
   c. Add matching sections to zero dispersion at ID straights.
   d. Match Twiss functions at straights for desired β functions.
4. Verify: emittance, tunes, chromaticity, beta functions, dispersion.

### Stage 2: Chromaticity Correction and Sextupole Optimization

1. Place sextupole families in -I pairs (chromatic sextupoles).
2. Set sextupole strengths to correct chromaticity to small positive values.
3. Add harmonic sextupoles to minimize RDTs and ADTS.
4. Iterate: track dynamic aperture (DA) with particle tracking, adjust
   sextupole strengths and positions.

### Stage 3: Dynamic Aperture and Momentum Acceptance Optimization

1. Track particles for ~1000 turns with varying initial amplitudes.
2. Map DA in (x, y) space at injection point.
3. Target: DA > 2–3 mm in x for off-axis injection; or > 0.5 mm for on-axis.
4. Scan off-momentum DA at δ = ±3% or more for Touschek lifetime.
5. Optimize with multi-objective algorithms (MOGA, particle swarm, etc.).

### Stage 4: Injection Scheme

- Off-axis injection (pulsed bump): needs DA > ~3 mm horizontally.
- On-axis injection (pulsed multipole or swap-out): relaxes DA requirement
  but needs fast kickers.
- Swap-out injection: extract stored bunch, inject fresh one — needs
  excellent kicker reproducibility.
- Longitudinal injection: uses large energy offset → needs momentum
  acceptance > 3%.

### Stage 5: Collective Effects and Impedance

1. Intrabeam scattering (IBS): dominant source of emittance growth in
   low-emittance rings. IBS growth rate ∝ N_particles / (ε^(3/2) · σ_z).
   Must estimate the equilibrium emittance including IBS.
2. Impedance budget: resistive wall, BPMs, bellows, RF cavities, IDs.
3. Instability thresholds: TMCI, coupled-bunch instabilities, microwave
   instability. Compare growth rates vs. radiation damping rates.
4. Touschek lifetime: inversely related to charge density. Low-emittance
   rings have short Touschek lifetimes → need top-up injection.


## Reference Machine Parameters

### Large High-Energy Rings (E ≥ 6 GeV)

ESRF-EBS:
- Energy: 6 GeV, Circumference: 844 m, Cells: 32
- Lattice: H7BA, Emittance: 133 pm·rad
- Current: 200 mA, 992 bunches

APS-U:
- Energy: 6 GeV, Circumference: 1104 m, Cells: 40
- Lattice: H7BA, Emittance: 42 pm·rad
- Current: 200 mA, 1296 bunches

HEPS:
- Energy: 6 GeV, Circumference: 1360.4 m, Cells: 48
- Lattice: H7BA, Emittance: 34.2 pm·rad
- Current: 200 mA, 680 bunches

### Mid-Energy Rings (2–4 GeV)

MAX IV (3 GeV Ring):
- Energy: 3 GeV, Circumference: 528 m, Cells: 20
- Lattice: 7BA, Emittance: 328 pm·rad
- Current: 500 mA

Sirius:
- Energy: 3 GeV, Circumference: 518.4 m, Cells: 20
- Lattice: 5BA, Emittance: 250 pm·rad
- Current: 350 mA

SLS-2:
- Energy: 2.4 GeV, Circumference: 290.4 m, Cells: 12
- Lattice: H7BA, Emittance: 100 pm·rad
- Current: 400 mA

SOLEIL-U:
- Energy: 2.75 GeV, Circumference: 354 m, Cells: 20
- Lattice: H7BA, Emittance: 52.5 pm·rad

Diamond-II:
- Energy: 3.5 GeV, Circumference: 561.6 m, Cells: 24
- Lattice: H6BA, Emittance: 160 pm·rad

### Compact Low-Energy Rings (2–2.5 GeV)

ALS-U:
- Energy: 2 GeV, Circumference: 196.5 m, Cells: 12
- Lattice: 9BA, Emittance: 108 pm·rad
- Current: 500 mA

HALF:
- Energy: 2.2 GeV, Circumference: 480 m, Cells: 24
- Lattice: H7BA, Emittance: 83 pm·rad

Elettra 2.0:
- Energy: 2.4 GeV, Circumference: 260 m, Cells: 12
- Lattice: H6BA, Emittance: 210 pm·rad

SKIF:
- Energy: 3 GeV, Circumference: 476 m, Cells: 16
- Lattice: H6BA, Emittance: 75 pm·rad


## Emittance Scaling Rules of Thumb

For storage rings based on multi-bend achromats:

    ε [pm·rad] ≈ F · E² [GeV²] / N_d³

where F is a lattice-dependent factor:
- F ≈ 1100–1600 for conventional MBA (non-optimized)
- F ≈ 500–800 for HMBA with LGB/RB
- F ≈ 300–500 for aggressively optimized HMBA

Practical guidance:
- For 3 GeV, 20 cells of 7BA (N_d = 140): ε ~ 200–400 pm·rad
- For 6 GeV, 48 cells of H7BA (N_d = 336): ε ~ 30–50 pm·rad
- For 2 GeV, 12 cells of 9BA (N_d = 108): ε ~ 100–200 pm·rad


## Momentum Compaction Factor

The momentum compaction factor α_c characterizes how the orbit length changes
with energy deviation:

    α_c = (1/C) ∮ η_x / ρ ds

For ultra-low-emittance rings: α_c is very small (~ 10⁻⁵ to 10⁻⁴) because
dispersion is minimized. Small α_c leads to:
- Short natural bunch length (~1 mm or less)
- Low microwave instability threshold
- Potential CSR issues

Trade-off: very small α_c makes beam more susceptible to longitudinal
instabilities.


## Energy Loss Per Turn and RF Considerations

Energy loss per turn from synchrotron radiation:

    U₀ = C_γ · E⁴ / ρ̄    with  C_γ = 8.85 × 10⁻⁵ m/GeV³

where ρ̄ is the average bending radius. For high-energy rings (6 GeV),
U₀ can reach several MeV, demanding multi-cell RF systems with total
voltages of 5–10 MV.

The RF voltage must satisfy:

    V_RF > U₀ / sin(φ_s)

where φ_s is the synchronous phase. The RF bucket height (maximum stable
energy deviation) depends on V_RF:

    δ_max ∝ √(V_RF / (α_c · h · E))


## Booster Ring Design Principles

When designing a booster ring as an injector for a storage ring:

1. The booster must accelerate from injection energy (typically 100–300 MeV
   from a linac) to the storage ring energy.
2. The lattice is usually a simple FODO, DBA, or TBA structure emphasizing
   large acceptance rather than low emittance.
3. Energy ramping: magnets ramp synchronously with the RF frequency. The
   ramping time is typically 50–500 ms.
4. The booster circumference should be a submultiple of the storage ring
   circumference for clean bunch transfer.
5. Key parameters: extraction emittance (must fit within storage ring DA),
   energy spread, bunch length at extraction.
6. Booster emittance at extraction should be small enough for efficient
   injection: typically < 10 nm·rad for off-axis injection into a 4th-gen
   storage ring.

## Straight Section Design

The ID (insertion device) straight sections must provide:
- Zero or near-zero dispersion (achromatic condition)
- Low and matched β functions (β_x typically 3–10 m, β_y typically 2–5 m)
- Sufficient length for undulators/wigglers (4–7 m typical)

Different straight lengths serve different purposes:
- Long straights (5–7 m): undulators, wigglers
- Medium straights (3–5 m): shorter IDs, diagnostics
- Short straights (~1–2 m): injection, RF cavities

The number of available ID straights directly determines the number of
beamlines the facility can support.
