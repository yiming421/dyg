"""Shared five-seed defaults and isolated workers for training entrypoints."""
from pathlib import Path
import subprocess
import sys

DEFAULT_SEEDS = (42, 43, 44, 45, 46)
DEFAULT_ROUTER_SEEDS = tuple(seed + 100 for seed in DEFAULT_SEEDS)


def add_seed_arguments(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--seed", type=int, help="Run one training seed at the given output path.")
    group.add_argument("--seeds", type=int, nargs="+", default=None,
                       help="Training seeds; defaults to 42 43 44 45 46 in separate seed directories.")


def launch_seed_workers(args, module, output_option, *, output_is_file=False):
    """Dispatch multi-seed training; explicit --seed keeps single-run behavior."""
    if args.seed is not None:
        return False
    seeds = list(DEFAULT_SEEDS if args.seeds is None else args.seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Training seeds must be nonempty and distinct")
    if getattr(args, "resume_from_checkpoint", None):
        raise ValueError("Resume one checkpoint with --seed and its original --output_dir")

    # Keep every caller option except the seed list and per-worker output path.
    argv = iter(sys.argv[1:])
    forwarded = []
    pending = None
    while True:
        token = pending if pending is not None else next(argv, None)
        pending = None
        if token is None:
            break
        option = token.split("=", 1)[0]
        if option == "--seeds":
            for value in argv:
                if value.startswith("--"):
                    pending = value
                    break
        elif option == output_option:
            if "=" not in token:
                next(argv)
        else:
            forwarded.append(token)

    output = Path(getattr(args, output_option[2:])).resolve()
    targets = [output.parent / f"seed-{seed}" / output.name if output_is_file
               else output / f"seed-{seed}" for seed in seeds]
    for target in targets:
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite {target}; choose a fresh output path")
    for seed, target in zip(seeds, targets):
        if output_is_file:
            target.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-m", module, *forwarded,
                   "--seed", str(seed), output_option, str(target)]
        print(f"Training seed {seed}: {target}", flush=True)
        subprocess.run(command, check=True)
    return True
