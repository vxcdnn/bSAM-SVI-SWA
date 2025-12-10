import argparse
import os
import pickle
import sys

import numpy as np
import torch
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from data import dataloader
from models import get_model
from util import normal_like_tree

num_workers = 0

def compute_ece_incremental(probs_list, labels_list, bins=20):
    bin_boundaries = np.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]  
    bin_sums = np.zeros(bins)
    bin_counts = np.zeros(bins)
    bin_accuracies = np.zeros(bins)
    
    # Process each batch
    for probs, labels in zip(probs_list, labels_list):
        probs_np = np.array(probs)
        labels_np = np.array(labels)
        
        confidences = np.max(probs_np, axis=1)
        predictions = np.argmax(probs_np, axis=1)
        accuracies = (predictions == np.argmax(labels_np, axis=1))
        
        for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
            in_bin_mask = (confidences >= bin_lower) & (confidences <= bin_upper)
            if np.any(in_bin_mask):
                bin_counts[i] += np.sum(in_bin_mask)
                bin_sums[i] += np.sum(confidences[in_bin_mask])
                bin_accuracies[i] += np.sum(accuracies[in_bin_mask].astype(float))
    
    ece = 0.0
    total_samples = np.sum(bin_counts)
    
    for i in range(bins):
        if bin_counts[i] > 0:
            avg_confidence = bin_sums[i] / bin_counts[i]
            accuracy = bin_accuracies[i] / bin_counts[i]
            prob_in_bin = bin_counts[i] / total_samples
            ece += np.abs(avg_confidence - accuracy) * prob_in_bin
    
    return ece

def compute_auroc_incremental(probs_list, labels_list, bins=1000):
    """Compute AUROC"""
    confidences = []
    is_correct = []
    
    for probs, labels in zip(probs_list, labels_list):
        probs_np = np.array(probs)
        labels_np = np.array(labels)
        
        conf = np.max(probs_np, axis=1)
        preds = np.argmax(probs_np, axis=1)
        truth = np.argmax(labels_np, axis=1)
        
        confidences.append(conf)
        is_correct.append(preds == truth)
    
    confidences = np.concatenate(confidences)
    is_correct = np.concatenate(is_correct)
    
    thresholds = np.linspace(1, 0, bins + 1)[1:-1]
    
    tprs = [0.0]
    fprs = [0.0]
    P = float(np.sum(is_correct))
    N = float(np.sum(~is_correct))
    
    for t in thresholds:
        above_threshold = confidences >= t
        if np.any(above_threshold):
            tp = float(np.sum(is_correct[above_threshold]))
            fp = float(np.sum(~is_correct[above_threshold]))
            
            tprs.append(tp / P if P > 0 else 0.0)
            fprs.append(fp / N if N > 0 else 0.0) 
    tprs.append(1.0)
    fprs.append(1.0)
    
    auroc = np.trapezoid(tprs, fprs)
    return auroc

def evaluate_batch(modelapply, params, state, batch_data, batch_targets, nclasses):
    dat = jnp.array(batch_data.numpy().transpose(0, 2, 3, 1))
    tgt = jax.nn.one_hot(batch_targets.numpy(), nclasses)
    
    logits, _ = modelapply(params, state, None, dat, is_training=False)
    probs = jax.nn.softmax(logits, axis=1)
    
    correct = jnp.sum(jnp.argmax(logits, axis=1) == jnp.argmax(tgt, axis=1))
    nll_value = -jnp.mean(jnp.sum(tgt * jax.nn.log_softmax(logits, axis=1), axis=1))
    
    return float(correct), float(nll_value), np.array(probs), np.array(tgt)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testbatchsize', type=int, default=128)
    parser.add_argument('--testmc', type=int, default=32)
    parser.add_argument('--resultsfolder', type=str, required=True)
    parser.add_argument('--use_swa', action='store_true')
    parser.add_argument('--max_batches', type=int, default=50, help='maximum batches to test')
    parser.add_argument('--skip_metrics', action='store_true', help='skip ECE and AUROC computation')
    parser.add_argument('--fast_metrics', action='store_true', help='use faster, approximate metrics')
    
    args = parser.parse_args()

    print(f"Loading from {args.resultsfolder}")
    
    with open(os.path.join(args.resultsfolder, 'trainstate.pickle'), 'rb') as file:
        trainstate = pickle.load(file)
        trainargs = pickle.load(file)
    
    randomseed = getattr(trainargs, 'randomseed', 42)
    rngkey = jax.random.PRNGKey(randomseed)
    np.random.seed(randomseed)
    torch.manual_seed(randomseed)
    
    augment_test = False
    trainset, testset, trainloader, testloader = dataloader(trainargs.dataset)(
        trainargs.batchsize, args.testbatchsize, 
        trainargs.datasetfolder, augment_test, num_workers
    )
    
    ndata = len(trainset)
    nclasses = len(trainset.classes)
    
    print(f"Dataset: {trainargs.dataset}, ntest={len(testset)}, nclasses={nclasses}")
    
    modelapply, modelinit = get_model(trainargs.model.lower(), nclasses)
    
    test_batch = next(iter(testloader))[0].numpy().transpose(0, 2, 3, 1)
    _, netstate = modelinit(rngkey, test_batch, True)
    
    if args.use_swa and 'swa_w_mean' in trainstate.optstate and trainstate.optstate['swa_n'] > 0:
        print(f"Using SWA average ({trainstate.optstate['swa_n']} snapshots)")
        theta = trainstate.optstate['swa_w_mean']
    else:
        print("Using current parameters")
        theta = trainstate.optstate['w']
    
    # Test with point estimate
    print(f"\n{'='*60}")
    print("1. Testing with point estimate (mean):")
    print(f"Testing on {args.max_batches} batches maximum")
    
    total_correct = 0
    total_samples = 0
    total_nll = 0.0
    all_probs = []
    all_labels = []
    batch_count = 0
    
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if batch_idx >= args.max_batches:
            break
            
        correct, nll_value, probs, labels = evaluate_batch(
            modelapply, theta, trainstate.netstate, inputs, targets, nclasses
        )
        
        total_correct += correct
        total_samples += inputs.shape[0]
        total_nll += nll_value
        
        if not args.skip_metrics:
            all_probs.append(probs)
            all_labels.append(labels)
        
        batch_count += 1
        
        if batch_idx % 5 == 0:
            current_acc = 100.0 * total_correct / total_samples
            print(f"   Batch {batch_idx}: {total_correct:.0f}/{total_samples} = {current_acc:.2f}%")
        
    testacc = 100.0 * total_correct / total_samples
    avg_nll = total_nll / batch_count if batch_count > 0 else 0.0
    
    print(f"\nBasic Results:")
    print(f"   > Accuracy: {testacc:.2f}%")
    print(f"   > NLL: {avg_nll:.4f}")
    
    if not args.skip_metrics and all_probs:
        print(f"\nComputing calibration metrics...")
        
        if args.fast_metrics:
            sample_probs = all_probs[:10]
            sample_labels = all_labels[:10]
            test_ece = compute_ece_incremental(sample_probs, sample_labels, bins=10)
            test_auroc = compute_auroc_incremental(sample_probs, sample_labels, bins=100)
        else:
            test_ece = compute_ece_incremental(all_probs, all_labels)
            test_auroc = compute_auroc_incremental(all_probs, all_labels)
        
        print(f"\n   Calibration Metrics:")
        print(f"   > ECE: {test_ece:.4f}")
        print(f"   > AUROC: {test_auroc:.4f}")    
    if args.testmc > 1:
        print(f"\n{'='*60}")
        print(f"2. Testing with Bayesian averaging ({args.testmc} samples):")       
        has_variance = False
        variance = None
        
        if 'swa_w_m2' in trainstate.optstate and trainstate.optstate['swa_n'] > 1:
            n = trainstate.optstate['swa_n']
            variance = jax.tree.map(
                lambda m2: m2 / jnp.maximum(n - 1, 1) + 1e-8,
                trainstate.optstate['swa_w_m2']
            )
            has_variance = True
            print("Using SWA variance for Bayesian sampling")
        elif 's' in trainstate.optstate:
            has_variance = True
            print("Using bSAM precision for Bayesian sampling")
        
        if has_variance:            
            bayes_correct = 0
            bayes_samples = 0
            bayes_nll = 0.0
            test_batches = 10
            
            for batch_idx, (inputs, targets) in enumerate(testloader):
                if batch_idx >= test_batches:
                    break
                
                dat = jnp.array(inputs.numpy().transpose(0, 2, 3, 1))
                tgt = jax.nn.one_hot(targets.numpy(), nclasses)
                
                sample_logits = []
                
                for i in range(args.testmc):
                    if variance is not None:
                        noise, rngkey = normal_like_tree(theta, rngkey)
                        theta_sampled = jax.tree.map(
                            lambda mu, var, n: mu + jnp.sqrt(var) * n,
                            theta, variance, noise
                        )
                    else:
                        noise, rngkey = normal_like_tree(theta, rngkey)
                        dafactor = getattr(trainargs, 'dafactor', 1.0)
                        theta_sampled = jax.tree.map(
                            lambda n, mu, s: mu + jnp.sqrt(1.0 / (ndata * dafactor * s)) * n,
                            noise, theta, trainstate.optstate['s']
                        )
                    
                    logits, _ = modelapply(theta_sampled, trainstate.netstate, None, dat, False)
                    sample_logits.append(logits)
                
                # Bayesian averaging
                temp = jax.nn.log_softmax(jnp.array(sample_logits), axis=2)
                bayes_logits = logsumexp(temp, b=1/args.testmc, axis=0)
                preds = jnp.argmax(bayes_logits, axis=1)
                
                bayes_correct += float(jnp.sum(preds == jnp.argmax(tgt, axis=1)))
                bayes_samples += dat.shape[0]
                bayes_nll += float(jnp.mean(jnp.sum(-tgt * bayes_logits, axis=1)))
                
                print(f"Batch {batch_idx}: complete")
            
            bayes_acc = 100.0 * bayes_correct / bayes_samples if bayes_samples > 0 else 0.0
            bayes_avg_nll = bayes_nll / test_batches
            
            print(f"\nBayesian Results:")
            print(f"   > Accuracy: {bayes_acc:.2f}%")
            print(f"   > NLL: {bayes_avg_nll:.4f}")
            print(f"   > Improvement: {bayes_acc - testacc:+.2f}% accuracy")
        else:
            print("Skipping Bayesian averaging (no variance information)")
if __name__ == '__main__':
    main()