# Follow-up acceptance regression

## Scope

This patch fixes a bounded-history acceptance inheritance defect. The old test
suite was intentionally removed; this is a new focused suite, not a restoration
or a claim that the former full suite has passed.

Before this change, only bare continuations such as `continue` inherited a
pending request's expert evidence requirements. Replies such as `b`, `Increase
the bump target to 40 mm.`, and `Continue, but keep the circumference fixed.`
could drop the pending `hybrid_mba_shape` requirement. A later `continue` after
an option reply also lost the original scope.

`continued_request_acceptance` now checks directly declared expert requirements
first, then recognizes common option replies and parameter amendments. It walks
through chains of these replies to the nearest pending request. Saved completion
markers and substantive new requests still stop inheritance. It does not edit
conversation history, accept any numerical result, or change physical limits.

An inherited evidence requirement is not permission to perform a requested
physical change; authorization and numerical acceptance remain separate.

## Run the new tests

From the repository root, with Python and pytest available:

```text
python -m pytest -q tests/test_scientific_acceptance_continuation.py
```

The tests load the actual scientific acceptance module and the actual MBA request
classifier directly from source. A small explicit registry double isolates them
from runtime setup. No model API, Julia installation, JuTrack execution, local
project archive, or restored legacy test fixture is needed.

Coverage includes option replies, target amendments, follow-up chains, a current
message already present in history, completion markers, unrelated inspections,
new expert scopes, empty history, history immutability, and rejection of missing
acceptance evidence after inheritance.

## Verification performed for this patch

Local Linux runtime: Python 3.13.5 and pytest 9.0.2.

| Source under test | Result with the same new suite |
| --- | --- |
| Original scientific_acceptance.py from 9edf4d5564c9880756d1fac0539d5f1b211f0532 | 23 failed, 32 passed |
| Patched scientific_acceptance.py | 55 passed |

The original source and unchanged MBA classifier were checked against their Git
blob hashes before the red/green run. The production patch and test file also
pass an AST syntax check using Python 3.10 grammar. That is not a Python 3.10
runtime test. Ruff was not available in the local validation environment.

No complete application, browser, Windows runtime, provider session, JuTrack
integration, or lattice-design campaign was run as part of this patch.

## Known limits and next boundary

This remains a conservative lexical fallback over the history supplied by the
caller. It is not a general semantic intent classifier. It can miss unrecognized
paraphrases, cannot recover an original request absent from the provided history,
and does not maintain several pending tasks across intervening unrelated requests.
A single-letter reply is treated as a potential option, not validated against a
specific menu. Ambiguous imperative topic changes still need explicit task-state
handling; passing this suite does not establish general language understanding.

A subsequent change should introduce an explicit persisted active-request identity
and amendment/supersession events, with provider/session integration tests. That is
separate from numerical H7BA method development. No H7BA design capability or physics
validation is claimed by this patch.
