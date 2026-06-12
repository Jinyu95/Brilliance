#!/usr/bin/env python
"""Placeholder — knowledge corpus build is no longer used.

The BRILLIANCE project no longer uses a knowledge corpus or graph.
This script is kept as a no-op so existing tooling that calls it does not break.
"""


def main() -> int:
    print("Knowledge corpus build is disabled (not used in JuTrack assistant mode).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
