import torch
import numpy as np
import time
import random
import itertools
from scipy.sparse.csgraph import shortest_path
import ot
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from pathlib import Path

# --- Helper functions from the original script (unchanged) ---
log_file_path = "log_analysis_multi.txt" # New log file
def logLine(s, verbose=True):
    if verbose:
        print(s)
    with open(log_file_path, "a") as log_file:
        log_file.write(str(s) + "\n")

def getGeodesics(P, K_conn, linkage=None): # Renamed K to K_conn for clarity
    t0 = time.time()
    D = torch.cdist(P, P)
    
    t0 = time.time()
    if K_conn <= 0: # K_conn is connectivity here
        M = torch.zeros_like(D, dtype=torch.bool) # No edges if K_conn is 0
    elif K_conn >= D.shape[1]: # If K_conn is full, connect all
        M = torch.ones_like(D, dtype=torch.bool)
        M.fill_diagonal_(False) # No self-loops in typical KNN graph
    else:
        # Ensure K_conn is not larger than the number of available neighbors (excluding self)
        actual_K_conn = min(K_conn, D.shape[1] - 1 if D.shape[1] > 1 else 0) 
        if actual_K_conn <=0: 
            M = torch.zeros_like(D, dtype=torch.bool) 
        else:
            # Set diagonal to infinity to prevent selecting self with topk
            D_no_self = D.clone()
            D_no_self.fill_diagonal_(torch.inf)
            _, indices = torch.topk(D_no_self, actual_K_conn, dim=1, largest=False, sorted=False)
            M = torch.zeros_like(D, dtype=torch.bool)
            M.scatter_(dim=1, index=indices, value=True)
    
    if linkage == 'mutual':
        M = M & M.T
    else: 
        M = M | M.T 
    
    D_graph = D.clone() 
    D_graph[~M] = torch.inf 
    D_graph.fill_diagonal_(0) # Shortest path to self is 0

    t0 = time.time()
    D_np = D_graph.cpu().numpy().astype(np.float64)
    G = shortest_path(D_np, method='auto', directed=False, unweighted=False, return_predecessors=False)
    
    # Handle disconnected components: replace inf with a large penalty
    # This makes C finite for OT_sampling, but disconnected points remain far.
    max_finite_dist = G[np.isfinite(G)].max() if np.any(np.isfinite(G)) else 1.0
    num_nodes = G.shape[0]
    G[G == np.inf] = max_finite_dist * num_nodes * 10 # Large penalty

    G[np.isnan(G)] = max_finite_dist * num_nodes * 10 # Also handle potential NaNs
    
    return torch.from_numpy(G).to(P.device)


def OT_sampling(k: int, X: np.ndarray, C: np.ndarray, max_iters: int = 100, num_sinkhorn_iter: int = 1000):
    N, d_features = X.shape if X.ndim > 1 else (X.shape[0], 0) # Handle X being 1D (e.g. N items, 0 features)

    if k == 0:
        return np.array([]), np.array([])
    
    effective_k = k
    if k > N :
        effective_k = N
        
    C_processed = np.ascontiguousarray(C, dtype=np.float64)
    
    # Check for all-infinite or all-NaN cost matrix (e.g., fully disconnected graph from getGeodesics)
    if not np.any(np.isfinite(C_processed)):
        # logLine(f"Warning: Cost matrix C for OT_sampling (k={k}) contains no finite values. Fallback to random.")
        rand_indices = np.random.choice(N, effective_k, replace=False)
        rand_weights = np.ones(effective_k) / effective_k
        return rand_indices, rand_weights

    if np.any(np.isinf(C_processed)) or np.any(np.isnan(C_processed)):
        max_finite_c = np.max(C_processed[np.isfinite(C_processed)], initial=0)
        C_processed[np.isinf(C_processed)] = max_finite_c * N * 100 # Increased penalty
        C_processed[np.isnan(C_processed)] = max_finite_c * N * 100

    cQ = np.random.choice(N, effective_k, replace=False)
    cQw = np.ones(effective_k) / effective_k
    Xw = np.ones(N) / N
    
    Cmax = np.max(C_processed) if np.any(C_processed) else 0.0
    if Cmax == 0 and np.all(C_processed == 0):
        final_Q_indices = np.random.choice(N, effective_k, replace=False)
        final_Qw = np.ones(effective_k) / effective_k
        return final_Q_indices, final_Qw
    
    C_scaled = C_processed / (Cmax + 1e-9) # Add epsilon to avoid division by zero if Cmax is 0

    Q_indices = cQ.copy()
    Qw_ = cQw.copy()

    for iteration_count in range(max_iters):
        cQC = C_scaled[:, cQ]
        cQw[cQw < 1e-9] = 1e-9 
        cQw = cQw / np.sum(cQw)

        try:
            T = ot.sinkhorn(
                a=Xw, b=cQw, M=cQC, reg=0.01,
                method='sinkhorn_stabilized', numItermax=num_sinkhorn_iter, warn=False 
            )
        except Exception as e:
            if iteration_count > 0: break 
            else:
                final_Q_indices = np.random.choice(N, effective_k, replace=False)
                final_Qw = np.ones(effective_k) / effective_k
                return final_Q_indices, final_Qw
        
        nextQk_indices = np.full(effective_k, -1, dtype=int)
        selected_indices_in_current_step = set()

        for j_cluster in range(effective_k):
            weights_for_barycenter = T[:, j_cluster]
            barycenter_candidate_costs = weights_for_barycenter @ C_scaled
            sorted_candidate_indices = np.argsort(barycenter_candidate_costs)
            found_unique_candidate_for_slot = False
            for candidate_idx in sorted_candidate_indices:
                if candidate_idx not in selected_indices_in_current_step:
                    nextQk_indices[j_cluster] = candidate_idx
                    selected_indices_in_current_step.add(candidate_idx)
                    found_unique_candidate_for_slot = True
                    break
            if not found_unique_candidate_for_slot:
                all_possible_indices = set(range(N))
                available_indices = list(all_possible_indices - selected_indices_in_current_step)
                if available_indices:
                    nextQk_indices[j_cluster] = available_indices[0]
                    selected_indices_in_current_step.add(available_indices[0])
                else: # Should not happen if effective_k <= N
                    if j_cluster < len(Q_indices) and Q_indices[j_cluster] not in selected_indices_in_current_step:
                         nextQk_indices[j_cluster] = Q_indices[j_cluster]
                    elif available_indices: # Should be caught by previous if, but for safety
                         nextQk_indices[j_cluster] = np.random.choice(available_indices)
                    else: # Absolute fallback, very unlikely
                         nextQk_indices[j_cluster] = np.random.choice(list(all_possible_indices - {Q_indices[j_cluster]})) if len(all_possible_indices - {Q_indices[j_cluster]}) > 0 else Q_indices[j_cluster]


        cQ = nextQk_indices
        nextQw = np.sum(T, axis=0)
        if np.sum(nextQw) > 1e-9 : nextQw = nextQw / np.sum(nextQw)
        else: nextQw = np.ones(effective_k) / effective_k 
        cQw = nextQw

        if np.array_equal(Q_indices, cQ): break
        Q_indices = cQ.copy()
        Qw_ = cQw.copy()
    return Q_indices, Qw_

# --- Analysis specific functions (calculate_estimated_fitnesses is mostly unchanged) ---
def calculate_estimated_fitnesses(gt_tr_population, selected_indices, weights):
    estimated_fitnesses = []
    actual_mean_fitnesses = []
    
    if not isinstance(selected_indices, np.ndarray) or selected_indices.size == 0:
        num_pop = gt_tr_population.shape[0]
        act_means = [gt_tr_population[i].mean().item() for i in range(num_pop)]
        return [np.nan] * num_pop, act_means

    weights_tensor = torch.tensor(weights, dtype=gt_tr_population.dtype, device=gt_tr_population.device)
    
    for i in range(gt_tr_population.shape[0]):
        individual_gt_tr = gt_tr_population[i, :]
        selected_indices_torch = torch.from_numpy(selected_indices.astype(np.int64)).to(individual_gt_tr.device)
        
        perf_on_subset = individual_gt_tr[selected_indices_torch]
        
        if weights_tensor.shape[0] != perf_on_subset.shape[0]:
             #This can happen if effective_k in OT_sampling was different from original k due to k > N
            if perf_on_subset.shape[0] > 0: # If some samples were selected
                 weights_tensor_adj = torch.ones(perf_on_subset.shape[0], dtype=gt_tr_population.dtype, device=gt_tr_population.device) / perf_on_subset.shape[0]
                 estimated_fitness = torch.dot(perf_on_subset, weights_tensor_adj)
            else: # No samples selected
                 estimated_fitness = torch.tensor(np.nan)
        elif perf_on_subset.nelement() == 0 : # no elements selected, e.g. k=0
            estimated_fitness = torch.tensor(np.nan)
        else:
            estimated_fitness = torch.dot(perf_on_subset, weights_tensor)

        estimated_fitnesses.append(estimated_fitness.item())
        actual_mean_fitnesses.append(individual_gt_tr.mean().item())
        
    return estimated_fitnesses, actual_mean_fitnesses


def process_single_file(data_file_path, K_VALUES_master):
    logLine(f"Processing single file: {data_file_path}")
    
    try:
        data = torch.load(data_file_path, map_location=torch.device('cpu'))
    except Exception as e:
        logLine(f"Error loading {data_file_path}: {e}. Skipping this file.")
        # Return structure with NaNs for all K_VALUES_master
        nan_results = {'times': [np.nan] * len(K_VALUES_master), 'pearsons': [np.nan] * len(K_VALUES_master)}
        return {method: nan_results.copy() for method in ["OT-PPS", "Random", "Stratified"]}

    if not data.get("GtTr") or len(data["GtTr"]) < 2:
        logLine(f"Error: GtTr[1] not available in {data_file_path}. Skipping.")
        nan_results = {'times': [np.nan] * len(K_VALUES_master), 'pearsons': [np.nan] * len(K_VALUES_master)}
        return {method: nan_results.copy() for method in ["OT-PPS", "Random", "Stratified"]}
        
    GtTr_gen1 = data["GtTr"][1].float()
    
    initial_embeddings_pop_x_train = data.get("embeddings")
    if initial_embeddings_pop_x_train is None or initial_embeddings_pop_x_train.numel() == 0:
        logLine(f"Error: 'embeddings' tensor is missing or empty in {data_file_path}. Skipping.")
        nan_results = {'times': [np.nan] * len(K_VALUES_master), 'pearsons': [np.nan] * len(K_VALUES_master)}
        return {method: nan_results.copy() for method in ["OT-PPS", "Random", "Stratified"]}

    P_dataset_items = initial_embeddings_pop_x_train.float().T
        
    n_train_items = GtTr_gen1.shape[1]
    if P_dataset_items.shape[0] != n_train_items:
        logLine(f"Warning in {data_file_path}: Mismatch num_train_items from GtTr_gen1 ({n_train_items}) and P_dataset_items ({P_dataset_items.shape[0]})")
        n_train_items = P_dataset_items.shape[0] # Prioritize P_dataset_items for sampling domain

    if n_train_items == 0:
        logLine(f"Error: Number of training items is 0 in {data_file_path}. Skipping.")
        nan_results = {'times': [np.nan] * len(K_VALUES_master), 'pearsons': [np.nan] * len(K_VALUES_master)}
        return {method: nan_results.copy() for method in ["OT-PPS", "Random", "Stratified"]}

    time_per_nll_eval = (data["tNLL"] / data["nNLL"]) if data.get("nNLL", 0) > 0 else 0

    file_results = {
        method: {'times': [np.nan] * len(K_VALUES_master), 'pearsons': [np.nan] * len(K_VALUES_master)}
        for method in ["OT-PPS", "Random", "Stratified"]
    }

    # --- OT-PPS ---
    # logLine(f"  OT-PPS for {data_file_path}...")
    geodesic_connectivity_K = int(n_train_items**0.5) if n_train_items > 0 else 0
    if P_dataset_items.shape[0] > 1 and geodesic_connectivity_K > 0 :
        C_geodesic = getGeodesics(P_dataset_items, geodesic_connectivity_K)
        C_geodesic_np = C_geodesic.cpu().numpy()
    elif P_dataset_items.shape[0] > 0 : # Fallback for very small P_dataset_items or K_conn=0
        C_geodesic = torch.cdist(P_dataset_items, P_dataset_items) 
        C_geodesic_np = C_geodesic.cpu().numpy()
    else: # P_dataset_items is empty or 1 item, C_geodesic_np cannot be formed meaningfully
        C_geodesic_np = np.array([[]])


    for k_idx, k_sampling in enumerate(K_VALUES_master):
        if k_sampling <= 0 or k_sampling > n_train_items or P_dataset_items.numel() == 0 or C_geodesic_np.size == 0:
            file_results["OT-PPS"]["times"][k_idx] = np.nan
            file_results["OT-PPS"]["pearsons"][k_idx] = np.nan
            continue
        
        t_start_sampling = time.time()
        X_ot = P_dataset_items.cpu().numpy()
        if X_ot.ndim == 1: X_ot = X_ot.reshape(-1,1) # ensure 2D for OT_sampling
        if X_ot.shape[1] == 0: X_ot = np.zeros((X_ot.shape[0], 1)) # ensure features if none

        selected_indices_ot, weights_ot = OT_sampling(k_sampling, X_ot, C_geodesic_np)
        time_sampling_algo = time.time() - t_start_sampling
        
        actual_k_selected = len(selected_indices_ot)
        total_time = (actual_k_selected * time_per_nll_eval) + time_sampling_algo
        
        est_fitness, act_fitness = calculate_estimated_fitnesses(GtTr_gen1, selected_indices_ot, weights_ot)
        
        pearson_r = np.nan
        if not np.isnan(est_fitness).all() and len(est_fitness) >= 2:
            valid_indices = ~np.isnan(est_fitness) & ~np.isnan(act_fitness)
            if np.sum(valid_indices) >= 2:
                est_f_clean = np.array(est_fitness)[valid_indices]
                act_f_clean = np.array(act_fitness)[valid_indices]
                if len(np.unique(est_f_clean)) >= 2 and len(np.unique(act_f_clean)) >= 2:
                    pearson_r, _ = pearsonr(est_f_clean, act_f_clean)
        
        file_results["OT-PPS"]["times"][k_idx] = total_time
        file_results["OT-PPS"]["pearsons"][k_idx] = pearson_r

    # --- Random Sampling ---
    # logLine(f"  Random for {data_file_path}...")
    for k_idx, k_sampling in enumerate(K_VALUES_master):
        if k_sampling <= 0 or k_sampling > n_train_items:
            file_results["Random"]["times"][k_idx] = np.nan
            file_results["Random"]["pearsons"][k_idx] = np.nan
            continue

        t_start_sampling = time.time()
        selected_indices_rand = np.array(random.sample(range(n_train_items), k_sampling))
        weights_rand = torch.ones(k_sampling) / k_sampling if k_sampling > 0 else torch.tensor([])
        time_sampling_algo = time.time() - t_start_sampling
        total_time = (k_sampling * time_per_nll_eval) + time_sampling_algo
        
        est_fitness, act_fitness = calculate_estimated_fitnesses(GtTr_gen1, selected_indices_rand, weights_rand.numpy())
        pearson_r = np.nan
        if not np.isnan(est_fitness).all() and len(est_fitness) >= 2:
            valid_indices = ~np.isnan(est_fitness) & ~np.isnan(act_fitness)
            if np.sum(valid_indices) >= 2:
                est_f_clean = np.array(est_fitness)[valid_indices]
                act_f_clean = np.array(act_fitness)[valid_indices]
                if len(np.unique(est_f_clean)) >= 2 and len(np.unique(act_f_clean)) >= 2:
                    pearson_r, _ = pearsonr(est_f_clean, act_f_clean)
        
        file_results["Random"]["times"][k_idx] = total_time
        file_results["Random"]["pearsons"][k_idx] = pearson_r

    # --- Stratified Sampling ---
    # logLine(f"  Stratified for {data_file_path}...")
    can_stratify = P_dataset_items.shape[1] > 0 and P_dataset_items.shape[0] > 0 # Need features and items
    if can_stratify:
        norms = torch.linalg.norm(P_dataset_items, axis=1)
        sorted_indices_by_norm = torch.argsort(norms)
    
    for k_idx, k_sampling in enumerate(K_VALUES_master):
        if not can_stratify or k_sampling <= 0 or k_sampling > n_train_items:
            file_results["Stratified"]["times"][k_idx] = np.nan
            file_results["Stratified"]["pearsons"][k_idx] = np.nan
            continue
        
        t_start_sampling = time.time()
        selected_indices_strat_list = []
        if k_sampling > 0 and n_train_items > 0 : # n_train_items check redundant due to outer check but good practice
            # Ensure k_sampling is not more than n_train_items for binning
            actual_k_for_binning = min(k_sampling, n_train_items)
            bin_edges = np.linspace(0, n_train_items, actual_k_for_binning + 1, dtype=int)
            for i in range(actual_k_for_binning):
                start_idx_in_sorted = bin_edges[i]
                end_idx_in_sorted = bin_edges[i+1]
                current_bin_original_indices = sorted_indices_by_norm[start_idx_in_sorted:end_idx_in_sorted]
                if len(current_bin_original_indices) > 0:
                    median_offset_in_bin = len(current_bin_original_indices) // 2
                    selected_indices_strat_list.append(current_bin_original_indices[median_offset_in_bin].item())
        
        selected_indices_strat = np.array(list(set(selected_indices_strat_list))) # Ensure unique
        actual_k_selected = len(selected_indices_strat)

        weights_strat = torch.ones(actual_k_selected) / actual_k_selected if actual_k_selected > 0 else torch.tensor([])
        time_sampling_algo = time.time() - t_start_sampling
        total_time = (actual_k_selected * time_per_nll_eval) + time_sampling_algo

        est_fitness, act_fitness = calculate_estimated_fitnesses(GtTr_gen1, selected_indices_strat, weights_strat.numpy())
        pearson_r = np.nan
        if not np.isnan(est_fitness).all() and len(est_fitness) >= 2:
            valid_indices = ~np.isnan(est_fitness) & ~np.isnan(act_fitness)
            if np.sum(valid_indices) >= 2:
                est_f_clean = np.array(est_fitness)[valid_indices]
                act_f_clean = np.array(act_fitness)[valid_indices]
                if len(np.unique(est_f_clean)) >= 2 and len(np.unique(act_f_clean)) >= 2:
                    pearson_r, _ = pearsonr(est_f_clean, act_f_clean)
        
        file_results["Stratified"]["times"][k_idx] = total_time
        file_results["Stratified"]["pearsons"][k_idx] = pearson_r
        
    return file_results


def run_multi_file_analysis(base_file_name="data_humaneval", num_files=5):
    logLine(f"Starting multi-file analysis for {base_file_name} (up to {num_files} files)")
    
    file_names = [f"{base_file_name}_{i}.pt" for i in range(num_files)]
    K_VALUES_master = list(range(4, 64)) 

    # Initialize accumulator for all runs
    # Structure: results_accumulator[method][k_value_master_idx]['pearsons_all_runs'] = [p_file0, p_file1, ...]
    #                                                            ['times_all_runs']    = [t_file0, t_file1, ...]
    methods = ["OT-PPS", "Random", "Stratified"]
    results_accumulator = {
        method: {
            k_idx: {'pearsons_all_runs': [], 'times_all_runs': []}
            for k_idx in range(len(K_VALUES_master))
        } for method in methods
    }

    processed_files_count = 0
    for file_idx, data_file_path_str in enumerate(file_names):
        data_file_path = Path(data_file_path_str)
        if not data_file_path.exists():
            logLine(f"File {data_file_path} not found. Skipping.")
            # Still need to append NaNs for this missing file to keep counts right for median over N files
            for method in methods:
                for k_idx in range(len(K_VALUES_master)):
                    results_accumulator[method][k_idx]['pearsons_all_runs'].append(np.nan)
                    results_accumulator[method][k_idx]['times_all_runs'].append(np.nan)
            continue
        
        processed_files_count +=1
        logLine(f"\n--- Processing available file {processed_files_count}: {data_file_path} ---")
        single_file_data = process_single_file(data_file_path, K_VALUES_master) # Renamed from single_file_results to avoid confusion
        
        # single_file_data is {method: {'times': [...], 'pearsons': [...]}}
        # where lists are indexed by K_VALUES_master
        for method in methods:
            if single_file_data and method in single_file_data: # Check if method processed
                for k_idx in range(len(K_VALUES_master)):
                    results_accumulator[method][k_idx]['pearsons_all_runs'].append(single_file_data[method]['pearsons'][k_idx])
                    results_accumulator[method][k_idx]['times_all_runs'].append(single_file_data[method]['times'][k_idx])
            else: # Method data not returned, fill with NaNs
                 for k_idx in range(len(K_VALUES_master)):
                    results_accumulator[method][k_idx]['pearsons_all_runs'].append(np.nan)
                    results_accumulator[method][k_idx]['times_all_runs'].append(np.nan)


    # Calculate medians
    median_results = {
        method: {'k_values': [], 'times': [], 'pearsons': []} for method in methods
    }

    for method in methods:
        for k_idx, k_val in enumerate(K_VALUES_master):
            pearsons_for_k = results_accumulator[method][k_idx]['pearsons_all_runs']
            times_for_k = results_accumulator[method][k_idx]['times_all_runs']
            
            # Only compute median if there's at least one non-NaN value
            if pearsons_for_k and not all(np.isnan(p) for p in pearsons_for_k):
                median_p = np.nanmedian(pearsons_for_k)
            else:
                median_p = np.nan

            if times_for_k and not all(np.isnan(t) for t in times_for_k):
                median_t = np.nanmedian(times_for_k)
            else:
                median_t = np.nan
            
            # Store k_val along with medians
            median_results[method]['k_values'].append(k_val)
            median_results[method]['times'].append(median_t)
            median_results[method]['pearsons'].append(median_p)
            
    # --- Plotting ---
    plt.figure(figsize=(12, 7)) # Slightly larger plot
    for method_name, method_data in median_results.items():
        # Filter out entries where median_t or median_p is NaN before plotting
        plot_k_values = []
        plot_times = []
        plot_pearsons = []
        for k_val, t, p in zip(method_data["k_values"], method_data["times"], method_data["pearsons"]):
            if not np.isnan(t) and not np.isnan(p):
                plot_k_values.append(k_val)
                plot_times.append(t)
                plot_pearsons.append(p)
        
        if plot_times: # Only plot if there's valid data
            # Sort by time for a clean line plot
            # Alternatively, sort by k_values if preferred, but time is on x-axis
            sorted_indices = np.argsort(plot_times)
            sorted_times = np.array(plot_times)[sorted_indices]
            sorted_pearsons = np.array(plot_pearsons)[sorted_indices]
            
            plt.plot(sorted_times, sorted_pearsons, marker='o', linestyle='-', label=f"{method_name}")
        else:
            logLine(f"No valid median data points to plot for method: {method_name}")

    plt.xlabel("Median Time Taken (s)")
    plt.ylabel("Median Pearson's Correlation to GtTr (Gen 1)")
    plt.title(f"Sampling Method Performance Comparison (Median over {processed_files_count} runs)\nDataset: {base_file_name} | Time vs. Pearson's R for Gen 1 GtTr Estimation")
    plt.legend(loc='best', fontsize='small')
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()
    
    plot_filename = f"analysis_median_{base_file_name}_time_vs_pearson_gen1.png"
    plt.savefig(plot_filename)
    logLine(f"Plot saved to {plot_filename}")
    # plt.show() # Comment out if running in a non-interactive environment

if __name__ == "__main__":
    with open(log_file_path, "w") as f: # Clear log file at start
        f.write("Starting Multi-File Analysis Log...\n")
    
    base_fn = "data_humaneval"
    num_f = 5

    # Create dummy files for demonstration if they don't exist
    for i in range(num_f):
        d_file = Path(f"{base_fn}_{i}.pt")
        if not d_file.exists():
            print(f"Creating dummy {d_file} for demonstration purposes.")
            # Make dummy data have slightly varying N_train_items to test robustness
            n_items_dummy = 50 + i * 5 
            pop_size_dummy = 10 + i
            dummy_data = {
                "GtTr": [
                    torch.randn(pop_size_dummy, n_items_dummy), 
                    torch.randn(pop_size_dummy, n_items_dummy) 
                ],
                "embeddings": torch.randn(pop_size_dummy, n_items_dummy), 
                "nNLL": float(500 + i * 50), 
                "tNLL": float(10.0 + i * 1.0) 
            }
            torch.save(dummy_data, d_file)
            logLine(f"Dummy file {d_file} created.")

    run_multi_file_analysis(base_file_name=base_fn, num_files=num_f)
    logLine("Multi-file analysis script finished.")