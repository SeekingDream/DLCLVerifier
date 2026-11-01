"""python -m dlclv status | validate"""
import collections, sys
from . import config


def main(argv):
    cmd = argv[1] if len(argv) > 1 else "status"
    if cmd == "validate":
        bad = config.validate()
        print("\n".join(bad) or "configs valid")
        return 1 if bad else 0
    if cmd == "status":
        c = collections.Counter()
        for kind, name, st in config.rows():
            c[(kind, st)] += 1
            print(f"{kind:<11}{st:<15}{name}")
        print("\nTOTAL", {f"{k}/{s}": n for (k, s), n in sorted(c.items())})
        return 0
    print(__doc__)
    return 2


sys.exit(main(sys.argv))
