from typing import List
import time
import random
import itertools
import torch
import math
import numpy as np
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.sparse.csgraph import shortest_path
from datasets import load_dataset
import ot

class PromptSample:
    count = 0
    def __init__(self, codons):
        self.codons = list(codons)
        self.fitness = None
        self.fitnessGtTr = None
        self.fitnessGtTe = None
        self.sampleId = PromptSample.count
        PromptSample.count+=1
    @property
    def prompt(self): return " ".join(self.codons)
    def __repr__(self):
        fitness_str = f"{self.fitness:.2f}" if self.fitness is not None else "None"
        fitnessTe_str = f"{self.fitnessGtTe:.2f}" if self.fitnessGtTe is not None else "None"
        return f"ID:{self.sampleId} fitness={fitness_str} fitnessTe={fitnessTe_str} prompt=<'{" ".join(self.codons)}'>"

log_file_path = "log.txt"
def logLine(s, verbose=False):
    if verbose:
        print(s)
    with open(log_file_path, "a") as log_file:
        log_file.write(str(s) + "\n")

def getGSM8k(n=None):
    dataset = []
    splits = ["train", "test"]
    for split in splits:
        gsm8k_data = load_dataset("gsm8k", "main", split=split)
        for sample in gsm8k_data:
            question = sample["question"]
            answer = sample["answer"]
            if "####" in answer:
                answer = answer.split("####", 1)[1].strip()
            dataset.append((question, answer))
    random.shuffle(dataset)
    if n and n < len(dataset):
        dataset = dataset[:n]
    return dataset

def getHumanEval(n=None):
    dataset = []
    humaneval = load_dataset("openai_humaneval")["test"]
    for sample in humaneval:
        prompt = sample["prompt"]
        canonical_solution = sample["canonical_solution"]
        dataset.append((prompt, canonical_solution))
    random.shuffle(dataset)
    if n and n<len(dataset):
        dataset = dataset[:n]
    return dataset

def getMMLU(n=None):
    dataset = []
    for split in ["test", "auxiliary_train"]:
        mmlu_data = load_dataset("cais/mmlu", "all")[split]
        for sample in mmlu_data:
            question = sample["question"]
            choices = [sample["choices"][i] for i in range(len(sample["choices"]))]
            formatted_question = question + "\n" + "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
            answer = [f"{chr(65+i)}" for i, _ in enumerate(choices)][sample["answer"]]
            dataset.append((formatted_question, answer, [f"{chr(65+i)}" for i, _ in enumerate(choices)]))
    random.shuffle(dataset)
    if n and n < len(dataset):
        dataset = dataset[:n]
    return dataset

def getCommonsenseQA(n=None):
    dataset = []
    splits = ["train", "validation"]
    for split in splits:
        cs_qa_data = load_dataset("commonsense_qa", split=split)
        for sample in cs_qa_data:
            question_text = sample["question"]
            if not (sample.get("choices") and sample["choices"].get("label") and sample["choices"].get("text")):
                continue
            choice_labels = sample["choices"]["label"]
            choice_texts = sample["choices"]["text"]
            if len(choice_labels) != len(choice_texts):
                continue
            formatted_choices = [f"{label}. {text}" for label, text in zip(choice_labels, choice_texts)]
            formatted_question = question_text + "\n" + "\n".join(formatted_choices)
            answer_key = sample["answerKey"]
            if not answer_key or answer_key not in choice_labels:
                continue
            dataset.append((formatted_question, answer_key, choice_labels))
    random.shuffle(dataset)
    if n and n < len(dataset):
        dataset = dataset[:n]
    return dataset

def getData(datasetID, n=None):
    datasetID=datasetID.lower()
    if datasetID=="humaneval":
        return getHumanEval(n)
    elif datasetID=="mmlu":
        return getMMLU(n)
    elif datasetID=="gsm8k":
        return getGSM8k(n)
    elif datasetID=="commonsenseqa":
        return getCommonsenseQA(n)
    else:
        logLine(f"Failed to fetch dataset, id not found - {datasetID}")
        return None

def getGeodesics(P, K, linkage=None):
    t0 = time.time()
    D = torch.cdist(P, P)
    logLine(f"t+{time.time()-t0:.2f}s Calculated Pairwise Dists")
    
    t0 = time.time()
    if K <= 0:
        return torch.zeros_like(D, dtype=torch.bool)
    if K >= D.shape[1]:
        return torch.ones_like(D, dtype=torch.bool)
    _, indices = torch.topk(D, K, dim=1, largest=False, sorted=False)
    M = torch.zeros_like(D, dtype=torch.bool)
    M.scatter_(dim=1, index=indices, value=True)
    if linkage == 'mutual':
        M = M & M.T
    else:
        M = M | M.T
    D[~M] = torch.inf
    logLine(f"t+{time.time()-t0:.2f}s Formed Graph")

    t0 = time.time()
    G = shortest_path(D.numpy(), method='auto', directed=False)
    logLine(f"t+{time.time()-t0:.2f}s Calculated Geodesics")
    return G

def OT_sampling(k: int, X: np.ndarray, C: np.ndarray, max_iters: int = 100, num_sinkhorn_iter: int = 1000):
    N, d_features = X.shape
    cQ = np.random.choice(N, k, replace=False)
    cQw = np.ones(k) / k
    Xw = np.ones(N) / N
    Cmax = np.max(C)
    if Cmax == 0:
        raise ValueError
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
                warn=False
            )
        except Exception as e:
            logLine(f"Warning: Sinkhorn failed in iteration {iteration_count + 1}: {e}")
            if iteration_count > 0:
                break 
            else:
                raise RuntimeError(
                    f"Sinkhorn algorithm failed on the first iteration: {e}. "
                    "Check input data, cost matrix, k, and regularization parameter 'reg'."
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
                    raise RuntimeError(
                        f"Failed to find a unique candidate for cluster slot {j_cluster} (k={k}, N={N}). "
                        "This should not happen if k <= N and N > 0."
                    )
        
        cQ = nextQk_indices
        
        nextQw = np.sum(T, axis=0)
        if np.sum(nextQw) > 1e-9 :
            nextQw = nextQw / np.sum(nextQw)
        else:
            logLine("Failure in weight update")
            nextQw = np.ones(k) / k
        cQw = nextQw

        if np.array_equal(Q_indices, cQ):
            logLine(f"Converged at iteration {iteration_count + 1}")
            break
        Q_indices = cQ.copy()
        Qw_ = cQw.copy()
        if iteration_count == max_iters -1:
            logLine(f"Reached max_iters {max_iters} without convergence of indices.")

    final_Q_indices = Q_indices
    final_Qw = Qw_
    
    return final_Q_indices, final_Qw

def getNLL(model, tokenizer, inputStr_list, outputStr_list, batch_size):
    all_neg_losses = []
    all_perplexities = []
    
    for i in range(0, len(inputStr_list), batch_size):
        batch_inputs = inputStr_list[i:i+batch_size]
        batch_outputs = outputStr_list[i:i+batch_size]
        
        combined_seq_list = [inp + out for inp, out in zip(batch_inputs, batch_outputs)]
        
        inputToks = tokenizer(combined_seq_list, return_tensors="pt", padding=True, truncation=True).to(model.device)
        batched_labels = inputToks.input_ids.clone()
        
        for j, input_str in enumerate(batch_inputs):
            input_len = len(tokenizer(input_str, add_special_tokens=False).input_ids)
            if tokenizer.bos_token_id is not None and inputToks.input_ids[j, 0] == tokenizer.bos_token_id:
                input_len += 1
            batched_labels[j, :input_len] = -100
            batched_labels[j, inputToks.input_ids[j] == tokenizer.pad_token_id] = -100
        
        with torch.no_grad():
            outputs = model(**inputToks, labels=batched_labels)
        
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = batched_labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss(reduction='none')
        
        for j in range(len(batch_inputs)):
            seq_len = inputToks.attention_mask[j].sum().item()
            loss_per_token = loss_fct(shift_logits[j, :seq_len-1].view(-1, shift_logits.size(-1)),
                                    shift_labels[j, :seq_len-1].view(-1))
            valid_losses = loss_per_token[shift_labels[j, :seq_len-1].view(-1) != -100]
            
            if len(valid_losses) > 0:
                item_loss = valid_losses.mean()
                all_neg_losses.append(-item_loss.item())
                all_perplexities.append(torch.exp(item_loss).item())
            else:
                all_neg_losses.append(float('-inf'))
                all_perplexities.append(float('inf'))
    
    return all_neg_losses, all_perplexities

def getPromptPerf(model, tokenizer, p, dataset, batch_size):
    meta = p + "\n"
    inputs = [meta + sample[0] + "\n" for sample in dataset]
    outputs = [sample[1] for sample in dataset]
    neg_logprobs, _ = getNLL(model, tokenizer, inputs, outputs, batch_size)
    probs_from_neg_logprobs = torch.tensor([math.exp(nlp) for nlp in neg_logprobs])
    return probs_from_neg_logprobs

def initPop(seedCodons: List[List[str]], k):
    codonSets = random.sample(list(itertools.product(*seedCodons)), k)
    return [PromptSample(codonSet) for codonSet in codonSets]

def reword(model, tokenizer, sentence: str) -> str:
    exampleOriginal = "The weather today is pleasant and sunny."
    exampleReworded = "It's a lovely, bright day today."
    prompt = (
        f"Original: {exampleOriginal}\n"
        f"Reworded: {exampleReworded}\n"
        f"Original: {sentence}\n"
        f"Reworded:" 
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    inputIds = inputs["input_ids"].to(model.device)
    attentionMask = inputs["attention_mask"].to(model.device)

    outputSequences = model.generate(
        input_ids=inputIds,
        attention_mask=attentionMask,
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        max_new_tokens=50,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    fullDecodedText = tokenizer.decode(outputSequences[0], skip_special_tokens=True)
    rewrittenPart = fullDecodedText[len(prompt):].split('\n')[0].strip()

    if not rewrittenPart:
        logLine("Could not generate a rewording.")
        rewrittenPart = sentence

    logLine(f"Reworded: <{sentence}> -> <{rewrittenPart}>")
    return rewrittenPart

def mutate(model, tokenizer, p1):
    codons = p1.codons[:]
    i = random.randint(0, len(codons) - 1)
    codons[i] = reword(model, tokenizer, codons[i])
    return PromptSample(codons)

def crossover(p1, p2):
    return PromptSample([random.choice(pair) for pair in zip(p1.codons, p2.codons)])

def main():
    global log_file_path
    logLine("Starting")
    modelName = "Qwen/Qwen2.5-7B"
    datasetID = "gsm8k"
    NINFBATCH = 4
    NDATASETRUNS = 5
    NPOP = 25
    NGENS = 10
    NELITES = 2
    SPLIT = 0.7
    seedCodons = [
        # List 1: Problem Introduction
        ["Solve:", "Calculate:", "Question:", "Problem:", "What is the answer to the following problem?", "The following is a math problem, provide the numerical solution:"],
        # List 2: Format Specification
        ["The numerical answer is", "The result is", "Solution:", "Value:", "", "The final number is:"],
        # List 3: Style
        ["Be precise.", "Focus on the numerical result.", "", "Output the number directly.", "Compute the exact value."]
    ]

    logLine("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(modelName)
    model = AutoModelForCausalLM.from_pretrained(modelName, torch_dtype=torch.float16, device_map="auto")
    logLine("Loaded model")

    for datasetrun in range(NDATASETRUNS):
        log_file_path = f"log_{datasetID}_{datasetrun}.txt"
        logLine(f"****Starting {datasetID}-{datasetrun}")
        
        logLine("Loading dataset...")
        fullDataset = getData(datasetID)
        splitIdx = int(len(fullDataset)*SPLIT)
        trainset = fullDataset[:splitIdx]
        testset = fullDataset[splitIdx:]
        logLine("Loaded dataset")

        logLine("Initializing probe prompts...")
        population = initPop(seedCodons, NPOP)
        embeddings_list = []
        logLine("Initialized probe prompts")

        logLine("Initializing embeddings")
        for s in population:
            sPerfs = getPromptPerf(model, tokenizer, s.prompt, trainset, NINFBATCH)
            s.fitnessGtTr = sPerfs.mean().item()
            s.fitness = s.fitnessGtTr
            sPerfs = getPromptPerf(model, tokenizer, s.prompt, testset, NINFBATCH)
            s.fitnessGtTe = sPerfs.mean().item()
            embeddings_list.append(sPerfs)
        embeddings = torch.stack(embeddings_list).T
        logLine("Initialized embeddings")
        # save embeddings to file

        logLine("Initializing manifold...")
        C = getGeodesics(embeddings,int(len(trainset)**0.5))
        logLine("Initialized manifold")
        
        for gen in range(NGENS):
            logLine(f"Starting Gen#{gen}")
            t0 = time.time()
            logLine(f"\tMutating")
            population += [mutate(model, tokenizer, random.choice(population[:NELITES])) for _ in range((NPOP - len(population))//2)]
            logLine(f"\tCrossovering")
            population += [crossover(*random.sample(population, 2)) for _ in range(NPOP - len(population))]
            logLine(f"\tEvaluating fitnesses")
            for s in population:
                if s.fitness is None and s.fitnessGtTr is None:
                    prompt_perfs_tensor = getPromptPerf(model, tokenizer, s.prompt, trainset, NINFBATCH)
                    s.fitnessGtTr = prompt_perfs_tensor.mean().item()
                    s.fitness = s.fitnessGtTr
                if s.fitnessGtTe is None:
                    prompt_perfs_tensor = getPromptPerf(model, tokenizer, s.prompt, testset, NINFBATCH)
                    s.fitnessGtTe = prompt_perfs_tensor.mean().item()

            population.sort(key=lambda s: s.fitness, reverse=True)
            tGen = time.time()-t0
            logLine(f"t+{tGen:.2f}s Finished Gen#{gen}")
            logLine([s.fitness for s in population])
            
            population = population[:NELITES]
            logLine(f"\tCurrent Best: {population[0]}")
        logLine(f"****Finished {datasetID}-{datasetrun}")

if __name__ == "__main__":
    main()