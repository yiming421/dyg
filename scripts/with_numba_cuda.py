#!/usr/bin/env python3
"""Run a Python script with the isolated DTGB Numba CUDA compiler libraries.

Usage from the repository root:
  python scripts/with_numba_cuda.py experiments/semantic_mlp/train_semantic_mlp_pipeline.py [original options]
CUDA_VISIBLE_DEVICES is preserved. The shared Python environment is not changed.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    repo=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda-home',type=Path,default=Path(os.environ.get('CUDA_HOME', str(repo/'outputs/runtime/numba_cuda_12_1'))))
    parser.add_argument('script',type=Path)
    parser.add_argument('args',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    cuda_home=args.cuda_home.resolve()
    nvvm=cuda_home/'nvvm/lib64'
    runtime=cuda_home/'lib64'
    if not (nvvm/'libnvvm.so').is_file():
        parser.error(f'Missing isolated NVVM library: {nvvm / "libnvvm.so"}')
    if not args.script.is_file():
        parser.error(f'Script does not exist: {args.script}')
    env=dict(os.environ,CUDA_HOME=str(cuda_home))
    paths=[str(nvvm),str(runtime)]
    if env.get('LD_LIBRARY_PATH'):paths.append(env['LD_LIBRARY_PATH'])
    env['LD_LIBRARY_PATH']=':'.join(paths)
    check=subprocess.run([sys.executable,'-c',
        'from numba import cuda; '
        'assert cuda.is_available(), "Numba CUDA is unavailable; refusing an unnoticed CPU fallback"; '
        'print("Numba CUDA ready:", cuda.get_current_device())'],env=env)
    if check.returncode:
        raise SystemExit(check.returncode)
    # Verify the actual source loaded by this launch, including kernel compilation.
    check=subprocess.run([sys.executable,str(repo/'tests/test_gpu_common_neighbors.py')],env=env,cwd=repo)
    if check.returncode:
        raise SystemExit(check.returncode)
    sys.stdout.flush()
    os.execve(sys.executable,[sys.executable,str(args.script),*args.args],env)


if __name__=='__main__':main()
