import torch
import numpy as np
import time
import random
import itertools
from scipy.sparse.csgraph import shortest_path
import ot
from scipy.stats import pearsonr # Note: pearsonr is imported but not used in the provided snippet
import matplotlib.pyplot as plt
from pathlib import Path # Note: Path is imported but not used in the provided snippet


def OT_sampling(k: int, X: np.ndarray, C: np.ndarray, max_iters: int = 100, num_sinkhorn_iter: int = 1000):
    N, d_features = X.shape
    cQ = np.random.choice(N, k, replace=False)
    cQw = np.ones(k) / k
    Xw = np.ones(N) / N
    Cmax = np.max(C)
    if Cmax == 0:
        # Handle cases where C is all zeros, e.g. if all points are identical
        # Or if K_geodesic was 0 leading to an empty graph and C of zeros.
        # Depending on desired behavior, could return random samples or raise error.
        # For now, let's assume C will be meaningful. If C is all zero, C/Cmax will fail.
        # A simple heuristic: if Cmax is very small (or zero), OT might not be meaningful.
        # Fallback to random sampling without weights, or just random indices with uniform weights.
        print(f"Warning: Cmax is {Cmax}. OT sampling might be unstable or meaningless.")
        if k > 0 and N > 0: # Ensure k and N are positive
            if N >= k:
                 # Fallback to random sampling if C is ill-conditioned
                print("Falling back to random sampling due to Cmax being zero or very small.")
                final_Q_indices = np.random.choice(N, k, replace=False)
                final_Qw = np.ones(k) / k
                return final_Q_indices, final_Qw
            else: # N < k, cannot sample k unique items
                 raise ValueError(f"Cannot sample k={k} items without replacement from N={N} items.")

        else: # k=0 or N=0
            return np.array([], dtype=int), np.array([], dtype=float)


    C = C / Cmax

    Q_indices = cQ.copy()
    Qw_ = cQw.copy()

    for iteration_count in range(max_iters):
        cQC = C[:, cQ]
        
        cQw[cQw < 1e-9] = 1e-9 
        cQw = cQw / np.sum(cQw)

        try:
            T = ot.sinkhorn(
                a=Xw,
                b=cQw,
                M=cQC,
                reg=0.01,
                method='sinkhorn_stabilized',
                numItermax=num_sinkhorn_iter,
                warn=False # Set to True if you want OT library warnings
            )
        except Exception as e:
            print(f"Warning: Sinkhorn failed in iteration {iteration_count + 1}: {e}")
            if iteration_count > 0:
                # If it failed after at least one successful iteration,
                # return the results from the last successful state.
                print("Returning results from before Sinkhorn failure.")
                break 
            else:
                # If it failed on the very first iteration, this is more critical.
                raise RuntimeError(
                    f"Sinkhorn algorithm failed on the first iteration: {e}. "
                    "Check input data (X, C), k, and regularization parameter 'reg'."
                ) from e
        
        nextQk_indices = np.full(k, -1, dtype=int)
        selected_indices_in_current_step = set()

        for j_cluster in range(k):
            weights_for_barycenter = T[:, j_cluster]
            barycenter_candidate_costs = weights_for_barycenter @ C
            
            sorted_candidate_indices = np.argsort(barycenter_candidate_costs)
            
            found_unique_candidate_for_slot = False
            for candidate_idx in sorted_candidate_indices:
                if candidate_idx not in selected_indices_in_current_step:
                    nextQk_indices[j_cluster] = candidate_idx
                    selected_indices_in_current_step.add(candidate_idx)
                    found_unique_candidate_for_slot = True
                    break
            
            if not found_unique_candidate_for_slot:
                # This case should ideally not be reached if k <= N and N > 0.
                # If it is, it might indicate an issue with the logic or input values.
                raise RuntimeError(
                    f"Failed to find a unique candidate for cluster slot {j_cluster} (k={k}, N={N}). "
                    "This might happen if k > N or due to issues in T matrix from Sinkhorn (e.g., all zeros)."
                )
        
        cQ = nextQk_indices
        
        nextQw = np.sum(T, axis=0)
        if np.sum(nextQw) > 1e-9 :
            nextQw = nextQw / np.sum(nextQw)
        else:
            print(f"Warning: Sum of nextQw is close to zero ({np.sum(nextQw)}). Resetting to uniform weights.")
            nextQw = np.ones(k) / k # Fallback to uniform weights if sum is too small
        cQw = nextQw

        if np.array_equal(Q_indices, cQ):
            # print(f"Converged at iteration {iteration_count + 1}")
            break
        Q_indices = cQ.copy()
        Qw_ = cQw.copy()
        if iteration_count == max_iters - 1:
            print(f"Reached max_iters {max_iters} without convergence of indices.")

    final_Q_indices = Q_indices
    final_Qw = Qw_
    
    return final_Q_indices, final_Qw

dataID = "humaneval"
i = 0
dataStr = f"data_{dataID}_{i}.pt"
data = torch.load(dataStr, map_location=torch.device('cpu'))
embeddings_np = data["embeddings"].numpy()


def MAE(est,gt):
    return torch.abs(est - gt).mean().item()

def sampleRnd(gt, k):
    nCols = gt.shape[1]
    if k == 0:
        return torch.tensor([], dtype=gt.dtype, device=gt.device).mean(dim=1) # Or handle as appropriate
    if k > nCols :
        k = nCols # cannot select more than available
    selIdx = torch.randperm(nCols)[:k]
    selCol = gt[:, selIdx]
    return selCol.mean(dim=1)


def getGeodesics(P, K, linkage=None):
    t0 = time.time()
    D = torch.cdist(P, P)
    print(f"t+{time.time()-t0:.2f}s Calculated Pairwise Dists")
    
    t0 = time.time()
    if K <= 0 :
        # Return a graph with no edges if K is non-positive
        # The shortest_path function might handle D values of torch.inf appropriately
        # Or return C as np.full_like(D.numpy(), np.inf)
        # For now, allow K=0 to effectively mean no nearest neighbors considered beyond self.
        # This leads to C where C[i,j]=inf if i!=j and C[i,i]=0
        G_no_edges = np.full(D.shape, np.inf, dtype=D.dtype.numpy())
        np.fill_diagonal(G_no_edges, 0)
        print(f"t+{time.time()-t0:.2f}s Formed Graph (K<=0, no edges beyond self-loops)")
        return G_no_edges

    if K >= D.shape[1]: # K is number of neighbors, D.shape[1] is N
        # If K is N or more, it's a fully connected graph (or effectively, as topk selects all)
        # In this case, D itself (after symmetrization if needed) can be the cost matrix.
        # However, shortest_path on D would just be D.
        M = torch.ones_like(D, dtype=torch.bool)
    else:
        _, indices = torch.topk(D, K, dim=1, largest=False, sorted=False)
        M = torch.zeros_like(D, dtype=torch.bool)
        M.scatter_(dim=1, index=indices, value=True)

    if linkage == 'mutual':
        M = M & M.T
    else: # Default to 'single' or 'OR' linkage behavior
        M = M | M.T
    
    D_graph = D.clone() # Clone D to avoid modifying the original D if it's used elsewhere
    D_graph[~M] = torch.inf 
    # Ensure diagonal is 0 for shortest_path
    torch.diagonal(D_graph).fill_(0)

    print(f"t+{time.time()-t0:.2f}s Formed Graph")

    t0 = time.time()
    G = shortest_path(D_graph.numpy(), method='auto', directed=False)
    # Handle inf values in G: these are disconnected components.
    # ot.sinkhorn can struggle with inf costs. Replace inf with a large number if necessary,
    # or ensure OT_sampling's C normalization handles it. Cmax=np.max(C) will be inf if G contains inf.
    # The C/Cmax normalization in OT_sampling should handle this (inf/inf -> nan, non-inf/inf -> 0).
    # It might be better to replace inf with a large finite number before passing to OT_sampling.
    if np.any(np.isinf(G)):
        print("Warning: Geodesic matrix C contains inf values (disconnected components). Replacing with large number.")
        max_finite_dist = np.max(G[np.isfinite(G)]) if np.any(np.isfinite(G)) else 1.0
        G[np.isinf(G)] = max_finite_dist * 10 # Replace inf with a value significantly larger than other distances
    print(f"t+{time.time()-t0:.2f}s Calculated Geodesics")
    return G

N_features = data["GtTr"][0].shape[1]
K_geodesic_neighbors = int(N_features**0.5)
C_cost_matrix_np = getGeodesics(data["embeddings"], K_geodesic_neighbors)


def sampleOT(gt_tensor, k_samples, current_embeddings_np, current_cost_matrix_np):
    if k_samples == 0:
        # Return an empty tensor or mean over zero elements, which might be NaN or error.
        # Consistent with sampleRnd, return mean of empty selection.
        # This would require careful handling in MAE or downstream.
        # For plotting, k usually starts from 1.
        num_prompts = gt_tensor.shape[0]
        return torch.full((num_prompts,), float('nan'), dtype=gt_tensor.dtype, device=gt_tensor.device)
    if k_samples > current_embeddings_np.shape[0]: # k cannot be greater than N
        print(f"Warning: k_samples ({k_samples}) > N ({current_embeddings_np.shape[0]}). Clamping k_samples to N.")
        k_samples = current_embeddings_np.shape[0]


    otSel_np, otW_np = OT_sampling(k_samples, current_embeddings_np, current_cost_matrix_np)

    if otSel_np.size == 0 and k_samples > 0 : # OT_sampling failed to return any samples for k > 0
        print(f"ERROR: OT_sampling returned empty selection for k={k_samples}. Check OT_sampling logic / inputs.")
        # Fallback or error handling
        num_prompts = gt_tensor.shape[0]
        return torch.full((num_prompts,), float('nan'), dtype=gt_tensor.dtype, device=gt_tensor.device)


    otSel_torch = torch.tensor(otSel_np, dtype=torch.long, device=gt_tensor.device)
    otW_torch = torch.tensor(otW_np, dtype=gt_tensor.dtype, device=gt_tensor.device)
    
    selCol = gt_tensor[:, otSel_torch]
    est = torch.matmul(selCol, otW_torch)
    return est

def sampleOT_wrapper(data_tuple, k_samples):
    gt_tensor, current_embeddings_np, current_cost_matrix_np = data_tuple
    return sampleOT(gt_tensor, k_samples, current_embeddings_np, current_cost_matrix_np)

from tqdm import tqdm
def sweep(fn,errs,ks,data_for_fn,gt_comparison,nRepeats=30):
    res = [[] for _ in range(len(errs))]
    for k_current in tqdm(ks):
        for i,errFn in enumerate(errs):
            cErrs = []
            for repeat_idx in range(nRepeats):
                # print(f"\tRepeat {repeat_idx+1}/{nRepeats}") # Optional: for verbose logging
                est = fn(data_for_fn, k_current)
                if torch.all(torch.isnan(est)): # Handle cases where est could be all NaNs (e.g. k=0)
                    cErrs.append(float('nan')) # MAE of NaN with anything is NaN
                else:
                    cErrs.append(errFn(est,gt_comparison))
            res[i].append(cErrs)
    return res

gen0 = data["GtTr"][0] 
gttr = gen0.mean(dim=1)

ks_for_sweep = list(range(1, min(64, N_features + 1)))

print("Starting Random Sampling Sweep...")
rndErrors = sweep(sampleRnd,[MAE],ks_for_sweep,gen0,gttr,nRepeats=1000)

print("\nStarting OT Sampling Sweep...")
data_for_ot_sweep = (gen0, embeddings_np, C_cost_matrix_np)
otErrors = sweep(sampleOT_wrapper,[MAE],ks_for_sweep,data_for_ot_sweep,gttr,nRepeats=100)


plt.figure(figsize=(10, 6))

def plot_mae_range(k_values, mae_data_series, color_plot, label_prefix):
    p25_values = []
    p50_values = []
    p75_values = []
    
    # mae_data_series[0] because we only have one error function (MAE)
    for errors_for_k in mae_data_series[0]:
        valid_errors = [e for e in errors_for_k if not np.isnan(e)]
        if not valid_errors: # If all errors were NaN for this k
            p25_values.append(np.nan)
            p50_values.append(np.nan)
            p75_values.append(np.nan)
        else:
            p25_values.append(np.percentile(valid_errors, 25))
            p50_values.append(np.percentile(valid_errors, 50))
            p75_values.append(np.percentile(valid_errors, 75))

    plt.plot(k_values, p50_values, label=f'{label_prefix} Median MAE (50th %ile)', color=color_plot)
    plt.fill_between(k_values, p25_values, p75_values, color=color_plot, alpha=0.2, label=f'{label_prefix} MAE 25th-75th %ile')
    

print("\nPlotting results...")
plot_mae_range(ks_for_sweep, rndErrors, 'blue', 'Random')
plot_mae_range(ks_for_sweep, otErrors, 'red', 'OT')

plt.xlabel('k (Number of Samples)')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('MAE vs. k for Random and OT Sampling')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

print("\nDone.")