import copy
import tempfile
from pathlib import Path
import numpy as np
from dynamics import contexts,context_tensor,replay,initial_prior,read_split,C

rng=np.random.default_rng(319)
data=rng.integers(0,5,size=(160,8),dtype=np.int16)
data[10:35:3,2]=-1
queries=[dict(query_id=f'q{i}',episode_id='e',machine_id=f'm{i}',machine_type='steady',cutoff=160) for i in range(8)]
ep=dict(data=data,queries=queries)
tensor=context_tensor(ep)
for t in (0,1,4,8,20,80,160):
    assert np.array_equal(tensor[t],contexts(data,t)),t
prior=np.tile([.5,.2,.15,.1,.05],(3,1))
config=dict(half_life=96,strength=2,temperature=.5,evidence_half_life=96)
first,_=replay(ep,prior,initial_prior(),config,[120])
altered=dict(data=data.copy(),queries=queries)
altered['data'][120:]=rng.integers(0,5,size=(40,8),dtype=np.int16)
second,_=replay(altered,prior,initial_prior(),config,[120])
assert np.array_equal(first[120],second[120]),'Same-tick/future leakage'
prefix=dict(data=data[:120].copy(),queries=queries)
third,_=replay(prefix,prior,initial_prior(),config)
assert np.array_equal(first[120],third[120]),'Historical forecast differs from prefix replay'
for p in first.values():
    assert np.all(np.isfinite(p)) and p.min()>=0 and np.allclose(p.sum(axis=1),1,atol=1e-12)
missing=dict(data=np.full((100,8),-1,dtype=np.int16),queries=queries)
p,_=replay(missing,prior,initial_prior(),config)
assert np.allclose(p[100],prior[0]),'Missing history must back off to prior'
with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
    path=Path(directory)
    (path/'queries.csv').write_text('query_id,episode_id,machine_id,machine_type,signal,cutoff\nq,e,m,steady,0,10\n')
    (path/'history.csv').write_text('episode_id,machine_id,tick,payout\ne,m,10,0\n')
    try:
        read_split(path)
        raise AssertionError('Future observation accepted')
    except ValueError:
        pass
print('PASS:',C,'structures; exact-lag vectorization; same-tick/future invariance; prefix equivalence; normalized factors; missing-data fallback; future-input rejection')
