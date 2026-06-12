# Distilled Accelerator Physics & Design Knowledge

This context is a release-safe distilled summary of the knowledge graph.
It is organized as learned knowledge, not as excerpts from any external source.

## Learned Domains
- JuTrack API: Optics & Analysis (Lattice, twissring, Number2TPSAD, ringpara): JuTrack API: Optics & Analysis (Lattice, twissring, Number2TPSAD, ringpara) is a learned cluster with 36 concepts and 51 detailed facts. Representative branches: Lattice, twissring, Number2TPSAD, ringpara, Beam.
- Beam Physics: energy, circumference, dynamic aperture: Beam Physics: energy, circumference, dynamic aperture is a learned cluster with 4 concepts and 33 detailed facts. Representative branches: energy, circumference, dynamic aperture, bunch length.
- Lattice Design: Multi-Bend Achromats (H7BA, 7BA, 9BA): Lattice Design: Multi-Bend Achromats (H7BA, 7BA, 9BA) is a learned cluster with 4 concepts and 10 detailed facts. Representative branches: H7BA, 7BA, 9BA, TME.
- Reference Machines: Diamond-II, SLS-2, APS-U: Reference Machines: Diamond-II, SLS-2, APS-U is a learned cluster with 6 concepts and 26 detailed facts. Representative branches: Diamond-II, SLS-2, APS-U, HEPS, SKIF.
- Beam Physics: emittance, Twiss: Beam Physics: emittance, Twiss is a learned cluster with 2 concepts and 27 detailed facts. Representative branches: emittance, Twiss.
- JuTrack API: Element Constructors (KQUAD, DRIFT, SBEND, KSEXT): JuTrack API: Element Constructors (KQUAD, DRIFT, SBEND, KSEXT) is a learned cluster with 15 concepts and 32 detailed facts. Representative branches: KQUAD, DRIFT, SBEND, KSEXT, findelem.
- Reference Machines: MAX IV, Elettra 2.0, HALF: Reference Machines: MAX IV, Elettra 2.0, HALF is a learned cluster with 6 concepts and 18 detailed facts. Representative branches: MAX IV, Elettra 2.0, HALF, SOLEIL-U, ALS-U.
- Lattice Design: off-axis injection, TPSA, on-axis injection: Lattice Design: off-axis injection, TPSA, on-axis injection is a learned cluster with 3 concepts and 4 detailed facts. Representative branches: TPSA, off-axis injection, on-axis injection.

## Key Themes
- Linepass / Copy Element / Deepcopy Element: Linepass / Copy Element / Deepcopy Element is a distilled topic group inside JuTrack API: Optics & Analysis (Lattice, twissring, Number2TPSAD, ringpara) with 25 concepts and 30 extracted details.
- Energy / Aperture / Dynamic: Energy / Aperture / Dynamic is a distilled topic group inside Beam Physics: energy, circumference, dynamic aperture with 3 concepts and 21 extracted details.
- Emittance / Twiss: Emittance / Twiss is a distilled topic group inside Beam Physics: emittance, Twiss with 2 concepts and 27 extracted details.
- H7Ba / Energy / Emittance: H7Ba / Energy / Emittance is a distilled topic group inside Lattice Design: Multi-Bend Achromats (H7BA, 7BA, 9BA) with 3 concepts and 8 extracted details.
- Circumference / Quantity: Circumference / Quantity is a distilled topic group inside Beam Physics: energy, circumference, dynamic aperture with 1 concepts and 12 extracted details.
- Ksext / Corrector / Hkicker: Ksext / Corrector / Hkicker is a distilled topic group inside JuTrack API: Element Constructors (KQUAD, DRIFT, SBEND, KSEXT) with 10 concepts and 12 extracted details.
- Diamond Ii / SLS-2 / SKIF: Diamond Ii / SLS-2 / SKIF is a distilled topic group inside Reference Machines: Diamond-II, SLS-2, APS-U with 3 concepts and 15 extracted details.
- Max / HALF / Soleil U: Max / HALF / Soleil U is a distilled topic group inside Reference Machines: MAX IV, Elettra 2.0, HALF with 3 concepts and 11 extracted details.
- Dipole / Combined Function: Dipole / Combined Function is a distilled topic group inside Magnets & Field Elements with 2 concepts and 3 extracted details.
- Injection / Off Axis / On Axis: Injection / Off Axis / On Axis is a distilled topic group inside Lattice Design: off-axis injection, TPSA, on-axis injection with 2 concepts and 3 extracted details.
- Ringpara / Getchrom / Gettune: Ringpara / Getchrom / Gettune is a distilled topic group inside JuTrack API: Optics & Analysis (Lattice, twissring, Number2TPSAD, ringpara) with 6 concepts and 11 extracted details.
- HEPS / Sirius / Emittance: HEPS / Sirius / Emittance is a distilled topic group inside Reference Machines: Diamond-II, SLS-2, APS-U with 2 concepts and 7 extracted details.

## JuTrack Concepts
- KQUAD (software_api): KQUAD (software_api) is linked to 7 extracted facts in learned cluster JuTrack API: Element Constructors (KQUAD, DRIFT, SBEND, KSEXT). Key details: jt.KQUAD(name, length, k1); jt.KQUAD("QF", 0.3, k1=1.5); jt.KQUAD("QD", 0.3, k1=-1.5)
- DRIFT (software_api): DRIFT (software_api) is linked to 5 extracted facts in learned cluster JuTrack API: Element Constructors (KQUAD, DRIFT, SBEND, KSEXT). Key details: jt.DRIFT(name, length); jt.DRIFT("D1", 0.5); jt.DRIFT("DL", 2.5)
- Lattice (software_api): Lattice (software_api) is linked to 5 extracted facts in learned cluster JuTrack API: Optics & Analysis (Lattice, twissring, Number2TPSAD, ringpara). Key details: jt.Lattice(elements_list); jt.Lattice([d1, qf, d2, bend, d3, qd, d4, bend,...
- set_field (software_api): set_field (software_api) is linked to 5 extracted facts in learned cluster JuTrack API: Utilities (set_field, DTPSAD, set_tps_dim, Gradient). Key details: jt.set_field(lat_tpsa[i], 'k1', k1_tpsa); jt.set_field(cell_tpsa[idx], 'k1', k1_tp...
- SBEND (software_api): SBEND (software_api) is linked to 4 extracted facts in learned cluster JuTrack API: Element Constructors (KQUAD, DRIFT, SBEND, KSEXT). Key details: jt.SBEND(name, length, angle); jt.SBEND("B", L_bend, angle_per_dipole); jt.SBEND("B", 1.5...
- twissring (software_api): twissring (software_api) is linked to 4 extracted facts in learned cluster JuTrack API: Optics & Analysis (Lattice, twissring, Number2TPSAD, ringpara). Key details: jt.twissring(lattice, dp=0.0, refpts=None); jt.twissring(ring_tpsa); jt....
- DTPSAD (software_api): DTPSAD (software_api) is linked to 3 extracted facts in learned cluster JuTrack API: Utilities (set_field, DTPSAD, set_tps_dim, Gradient). Key details: jt.DTPSAD(k1_value, var_idx); jt.DTPSAD(k1_vec[i], i+1); jt.DTPSAD(k1[i], i+1)
- KSEXT (software_api): KSEXT (software_api) is linked to 3 extracted facts in learned cluster JuTrack API: Element Constructors (KQUAD, DRIFT, SBEND, KSEXT). Key details: jt.KSEXT(name, length, k2); jt.KSEXT("SF", 0.15, k2=50.0); jt.KSEXT("SD", 0.15, k2=-50.0)
- Number2TPSAD (software_api): Number2TPSAD (software_api) is linked to 3 extracted facts in learned cluster JuTrack API: Optics & Analysis (Lattice, twissring, Number2TPSAD, ringpara). Key details: jt.Number2TPSAD(jt.Lattice(cell); jt.Number2TPSAD(cell.deepcopy(); jt...
- findelem (software_api): findelem (software_api) is linked to 3 extracted facts in learned cluster JuTrack API: Element Constructors (KQUAD, DRIFT, SBEND, KSEXT). Key details: jt.findelem(lattice, element_type=jt.element_types.KQUAD); jt.findelem(lattice, name="...
- ringpara (software_api): ringpara (software_api) is linked to 3 extracted facts in learned cluster JuTrack API: Optics & Analysis (Lattice, twissring, Number2TPSAD, ringpara). Key details: jt.ringpara(lattice, energy=3e9); jt.ringpara(lattice, energy=3e9, Vrf=3e...
- set_tps_dim (software_api): set_tps_dim (software_api) is linked to 3 extracted facts in learned cluster JuTrack API: Utilities (set_field, DTPSAD, set_tps_dim, Gradient). Key details: jt.set_tps_dim(n); jt.set_tps_dim(len(k1_vec); jt.set_tps_dim(2)

## Physics and Lattice Concepts
- emittance (physics_quantity): emittance (physics_quantity) is linked to 26 extracted facts in learned cluster Beam Physics: emittance, Twiss. Key details: Key parameters: extraction emittance (must fit within storage ring DA),; Booster emittance at extraction should...
- energy (physics_quantity): energy (physics_quantity) is linked to 18 extracted facts in learned cluster Beam Physics: energy, circumference, dynamic aperture. Key details: energy = 6 GeV; energy = 2.4 GeV; energy = 3.5 GeV
- circumference (physics_quantity): circumference (physics_quantity) is linked to 12 extracted facts in learned cluster Beam Physics: energy, circumference, dynamic aperture. Key details: circumference = 844 m; circumference = 1104 m; circumference = 1360.4 m
- dynamic aperture (physics_quantity): dynamic aperture (physics_quantity) is linked to 2 extracted facts in learned cluster Beam Physics: energy, circumference, dynamic aperture. Key details: dynamic_aperture = 3 mm; dynamic_aperture = 3 mm
- H7BA (lattice_type): H7BA (lattice_type) is linked to 4 extracted facts in learned cluster Lattice Design: Multi-Bend Achromats (H7BA, 7BA, 9BA). Key details: energy = 3 GeV; energy = 6 GeV; energy = 6 GeV
- Twiss (physics_quantity): Twiss (physics_quantity) is linked to 1 extracted facts in learned cluster Beam Physics: emittance, Twiss. Key details: requires Twiss
- bunch length (physics_quantity): bunch length (physics_quantity) is linked to 1 extracted facts in learned cluster Beam Physics: energy, circumference, dynamic aperture. Key details: bunch_length = 1 mm
- 7BA (lattice_type): 7BA (lattice_type) is linked to 2 extracted facts in learned cluster Lattice Design: Multi-Bend Achromats (H7BA, 7BA, 9BA). Key details: energy = 3 GeV; emittance = 400 rad
- 9BA (lattice_type): 9BA (lattice_type) is linked to 2 extracted facts in learned cluster Lattice Design: Multi-Bend Achromats (H7BA, 7BA, 9BA). Key details: energy = 2 GeV; emittance = 200 rad
- TME (lattice_type): TME (lattice_type) is linked to 2 extracted facts in learned cluster Lattice Design: Multi-Bend Achromats (H7BA, 7BA, 9BA). Key details: requires Twiss; requires dipole
- dipole (magnet_type): dipole (magnet_type) is linked to 2 extracted facts in learned cluster Magnets & Field Elements. Key details: If using reverse bends, the main dipole angles must compensate:; requires dipole
- TPSA (design_method): TPSA (design_method) is linked to 1 extracted facts in learned cluster Lattice Design: off-axis injection, TPSA, on-axis injection. Key details: enables Gradient
- off-axis injection (injection_method): off-axis injection (injection_method) is linked to 2 extracted facts in learned cluster Lattice Design: off-axis injection, TPSA, on-axis injection. Key details: dynamic_aperture = 3 mm; emittance = 10 rad
- combined-function dipole (magnet_type): combined-function dipole (magnet_type) is linked to 1 extracted facts in learned cluster Magnets & Field Elements. Key details: enables emittance
