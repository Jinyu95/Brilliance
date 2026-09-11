# Brilliance basic application contract

This contract is deliberately independent of accelerator-physics workflows. A TME, MBA, tracking,
impedance, or error study may use different scientific logic, but every Brilliance request must pass
through the same five application boundaries.

## 1. Runtime boundary

Brilliance—not the language model—owns Python, Julia, and JuTrack selection. Before the chat server
starts, an isolated probe must confirm all of the following:

- the configured Python is active;
- the configured `pyJuTrack` module loads;
- the JuTrack operations used by the public API are callable;
- Brilliance can construct canonical drift, quadrupole, sextupole, and bend elements;
- `findm66(ring, dp=0.0)` returns a finite 6 × 6 map;
- `findelem(ring, name=...)` observes the documented keyword and 1-based-index contract.

The versioned result is stored in `.venv/julia_env/brilliance-runtime.json`. A missing, stale, or
failed result prevents the normal launcher from accepting research work. Run `brilliance --check`
to execute and display this gate explicitly.

## 2. Program boundary

The model writes an ordinary named Python program under the current project's
`workspace/programs/`. The program uses `brilliance.accelerator_api`; it does not choose an
interpreter, discover JuTrack, or construct raw pyJuTrack elements.

Before execution, Brilliance parses the source and rejects known API mistakes, including direct
`pyJuTrack` imports, raw JuTrack element constructors, positional element-name use in `findelem`,
and incorrect `twissring`, `gettune`, `getchrom`, or `ringpara` result handling.

## 3. Execution boundary

Only the registered program runner starts scientific code. It snapshots the exact source and
arguments, uses the configured process environment, bounds or durably submits the calculation, and
captures stdout and stderr. A model shell fragment is never scientific evidence.

## 4. Evidence boundary

A program must call `open_research_run().finish(...)`. Brilliance accepts only the typed result:

- a plain-language summary;
- structured scalar data and real tables;
- declared plot files and downloads;
- an optional canonical lattice definition;
- `scientific_passed`, which describes calculation validity rather than goal attainment;
- `request_satisfied`, which is true only when this result answers the current chat request.

No `result.json` means no result. Invalid paths, missing plots, malformed tables, or an unexecutable
lattice fail deterministically without changing the working scientific state.

## 5. Conversation boundary

Chat is the product interface. Valid typed evidence is rendered automatically as prose, tables, and
inline plots. Source and JSON remain available for reproducibility but are not substitutes for an
answer. If a turn ends before execution, the chat states whether no program was created, a draft was
unfinished, a program was never launched, or a launched program failed to publish a typed result.

## Acceptance criterion

The basic application is usable only when this chain works end to end:

`configured runtime → saved program → JuTrack call → typed result → chat presentation`

Scientific expertise may extend what the program does, but it may not bypass or redefine this chain.
