"""GPU regression tests: duplicate groups, strict time cutoffs, CN/AA/RA."""
import json
import sys
from pathlib import Path
import numpy as np
from types import SimpleNamespace
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.modules import heuristic_models as hm
from utils.utils import get_neighbor_sampler


def make_sampler(src, dst, times):
    data=SimpleNamespace(src_node_ids=np.asarray(src,dtype=np.int64),
        dst_node_ids=np.asarray(dst,dtype=np.int64),
        node_interact_times=np.asarray(times,dtype=np.float64),
        edge_ids=np.arange(len(src)))
    return get_neighbor_sampler(data,'recent',seed=43)


def oracle(sampler,sources,targets,pred_times,mode):
    ptr,ids,times=hm.build_csr_from_neighbor_sampler(sampler)
    n=len(ptr)-1
    scores=[]
    for src,dst,t in zip(sources,targets,pred_times):
        if not (0<=src<n and 0<=dst<n):
            scores.append(0.);continue
        a=slice(ptr[src],ptr[src+1]);b=slice(ptr[dst],ptr[dst+1])
        common=set(ids[a][times[a]<t]) & set(ids[b][times[b]<t])
        value=0.
        for z in common:
            if mode=='cn':value+=1.
            else:
                degree=int(np.count_nonzero(times[ptr[z]:ptr[z+1]]<t))
                if mode=='ra' and degree>0:value+=1./degree
                if mode=='aa' and degree>1:value+=1./np.log(degree)
        scores.append(value)
    return np.asarray(scores)


def run():
    assert hm.HAS_CUDA,'These tests must execute the actual CUDA kernel'
    cases=[]
    tiny=make_sampler([1,1,2,2],[3,3,3,3],[1.,2.,1.,2.])
    cases.append(('repeated_neighbor',tiny,np.array([1,1,1,1,1,-1,1,3,0]),
        np.array([2,2,2,2,2,2,9,3,2]),np.array([0.,1.,1.5,2.,3.,3.,3.,3.,3.])))
    # One duplicate group spans many CUDA threads, with adversarial time order.
    big=make_sampler([1]*300+[2]*400+[4,5],[3]*700+[5,6],
        ([8.,1.,3.,2.,7.]*140)+[1.,2.])
    s=np.array([1,2,1,2,1,2,1,4,6,0,99,-1])
    d=np.array([2,1,2,1,1,2,3,6,4,2,2,2])
    t=np.array([1.,1.,2.,2.,3.,3.,9.,3.,3.,9.,9.,9.])
    cases.append(('large_duplicate_groups',big,s,d,t))
    rng=np.random.RandomState(20260919)
    sampler=make_sampler(rng.randint(1,16,1800),rng.randint(1,16,1800),rng.randint(0,6,1800).astype(float))
    s=rng.randint(-1,19,1000);d=rng.randint(-1,19,1000)
    t=rng.choice(np.array([-1.,0.,.5,1.,2.,3.,4.,5.,6.,10.]),1000)
    cases.append(('random_multigraph',sampler,s,d,t))
    checks=[]
    for name,sampler,s,d,t in cases:
        for mode in ['cn','aa','ra']:
            expected=oracle(sampler,s,d,t,mode)
            cpu=hm.score_links_by_common_neighbors(sampler,s,d,t,mode=mode,use_gpu=False)
            gpu=hm.score_links_by_common_neighbors(sampler,s,d,t,mode=mode,use_gpu=True)
            np.testing.assert_allclose(cpu,expected,rtol=1e-10,atol=1e-12)
            np.testing.assert_allclose(gpu,expected,rtol=1e-10,atol=1e-12)
            # Repeated invocation also catches stale accumulation/output buffers.
            gpu2=hm.score_links_by_common_neighbors(sampler,s,d,t,mode=mode,use_gpu=True)
            np.testing.assert_allclose(gpu2,expected,rtol=1e-10,atol=1e-12)
            checks.append(dict(case=name,mode=mode,queries=len(s),
                max_abs_error=float(np.max(np.abs(gpu-expected)))))
    empty=np.empty(0,dtype=np.int64)
    assert hm.score_links_by_common_neighbors(tiny,empty,empty,empty.astype(float),mode='ra',use_gpu=True).shape==(0,)
    return checks


def test_gpu_cpu_equivalence():
    if not hm.HAS_CUDA:
        import pytest
        pytest.skip('Requires real Numba CUDA; use scripts/with_numba_cuda.py')
    run()


if __name__=='__main__':print(json.dumps(run(),indent=2))
