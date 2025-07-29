import time
import numpy as np
import random
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path
from joblib import Memory
import juliacall
from juliacall import Main as jl
import torch
from functools import wraps
import dask
from dask.cache import Cache
import umap
import ot
pass

cachedir = './cache'
memory = Memory(cachedir, verbose=0)
def timeCache(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        @memory.cache
        def cachedFunc(*args, **kwargs):
            t0 = time.time()
            res = func(*args, **kwargs)
            return res, time.time() - t0
        return cachedFunc(*args, **kwargs)
    return wrapper

@dask.delayed
@timeCache
def getData(dataID, i):
    dataStr = f"data/data_{dataID}_{i}.pt"
    return torch.load(dataStr, map_location=torch.device('cpu'), weights_only=True)["embeddings"]

@dask.delayed
@timeCache
def getDist(P,metric):
    if metric=="euclidean": return torch.cdist(P, P).cpu().numpy()
    if metric=="cosine":
        normP=torch.nn.functional.normalize(P,p=2,dim=1,eps=1e-12)
        return 1-(normP@normP.T).cpu().numpy()

@dask.delayed
@timeCache
def getGraphKNN(D, K, linkage):
    D_=D.copy()
    indices = np.argpartition(D_, K, axis=1)[:, :K]
    M = np.zeros_like(D_, dtype=bool)
    np.put_along_axis(M, indices, True, axis=1)
    if linkage == 'mutual': M = M & M.T
    else: M = M | M.T
    D_[~M] = 0
    return csr_matrix(D_)

@dask.delayed
@timeCache
def getGraphEpsBall(D):
    D_ = D.copy()
    mst = minimum_spanning_tree(D_)
    M = (D_ <= mst.data.max() if mst.nnz > 0 else 0)
    D_[~M] = 0
    return csr_matrix(D_)

@dask.delayed
@timeCache
def getGraphUMAP(P, K):
    P_ = np.ascontiguousarray(P ,dtype=np.float64)
    umapLearner = umap.UMAP(n_neighbors=K, random_state=0)
    _ = umapLearner.fit(P_)
    graph = umapLearner.graph_
    graph.data = -np.log(graph.data)
    return graph

@dask.delayed
@timeCache
def getGeodesicsAPSP(G):
    return shortest_path(G, method='D', directed=False)

jl.seval("using Laplacians")
@dask.delayed
@timeCache
def getGeodesicsR(G):
    C=G.copy()
    C.data=1/(C.data+1e-9)
    return jl.Laplacians.effectiveResistances(C)

@dask.delayed
@timeCache
def getIdxRnd(P, k):
    return torch.randperm(P.shape[0])[:k]

@dask.delayed
@timeCache
def getIdxStratified(P, k):
    return (P**2).sum(1).argsort()[torch.linspace(0,P.shape[0]-1,k+2,device=P.device)[1:-1].long()]

@dask.delayed
@timeCache
def getIdxOT(C, k, maxIters=10, maxItersOT=100):
    C_ = C.copy()/C.max()
    N = C.shape[0]
    selIdx = np.random.choice(N, k, replace=False)
    selW = np.full(k,1/k)
    tgtW = np.full(N,1/N)
    for i in range(maxIters):
        try: T = ot.sinkhorn(a=tgtW, b=selW, C=C_[:, selIdx], reg=0.01, numItermax=maxItersOT).T
        except Exception as e: print(f"Sinkhorn failed at iteration {i}: {e}. Returning last valid result."); break
        selIdx_ = (T@C_).argmin(axis=1)
        if len(set(selIdx_)) != selIdx_.shape[0]:
            seen, firstIdx = np.unique(selIdx_, return_index=True)
            seen = set(seen)
            mask = np.ones_like(selIdx_, dtype=bool)
            mask[firstIdx] = False
            for i in np.where(mask)[0]:
                while((newVal:=np.random.randint(N)) in seen): pass
                seen.add(newVal)
                selIdx_[i]=newVal
        if set(selIdx)==set(selIdx_): break
        selIdx = selIdx_
        selW = np.sum(T,axis=1)
    return selIdx, selW

@dask.delayed
@timeCache
def MAE(gtPerf, selIdx, selW=None):
    selW = np.full(selIdx.shape[0],1/selIdx.shape[0]) if selW is None else selW
    selW /= selW.sum()
    sel = gtPerf[selIdx, :]
    estPerf = selW@sel
    return np.mean(np.abs(estPerf-gtPerf))

data = getData("humaneval",0)
i = getIdxStratified(data, 5)

pass