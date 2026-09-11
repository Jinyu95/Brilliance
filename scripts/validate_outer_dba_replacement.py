"""Verify historical outer-DBA evidence; new searches use the optical-contract script.

The old peak-in-quadrupole/free-core-optics experiment is not a valid conversion
route. Its archived numerical files remain readable and independently verifiable.
"""
import argparse
import json
import os
from pathlib import Path

from brilliance import accelerator_api as api
from brilliance.research_environment import ResearchEnvironment


def save_record(target, value):
    temporary = target.with_suffix(target.suffix+'.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)


def verify_saved(output):
    candidate = api.load_lattice(output/'candidate-0.json')
    result = api.evaluate(candidate, include_radiation=True, include_profile=False)
    bends = api.bend_radiation_contributions(candidate)
    report = dict(stable=result.linear_optics.stable, radiation=result.require_radiation(),
                  per_bend_consistency=bends['consistency'], per_bend=bends,
                  limitation='Historical numerical verification, not current hybrid acceptance')
    save_record(output/'full_ring_verification.json', report)
    print(json.dumps({key:report[key] for key in ('stable','radiation','per_bend_consistency')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    args, historical_args = parser.parse_known_args()
    if not args.verify_only:
        parser.error('New searches must use scripts/validate_hybrid_optical_contract.py; the old peak-only/free-optics route was retired.')
    env = ResearchEnvironment.load()
    env.require_active_python(operation='historical outer DBA verification')
    if env.internet_access:
        raise ValueError('Verification requires research internet disabled')
    verify_saved(args.output)


if __name__ == '__main__':
    main()
