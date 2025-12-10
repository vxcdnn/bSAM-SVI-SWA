import argparse
import os
import sys
import pickle

import numpy as np
import torch
from tqdm import trange, tqdm

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import haiku as hk

from data import dataloader
from models import get_model
from util import tprint, nll_categorical, normal_like_tree
from optim import (
    build_sgd_optimizer, 
    build_sam_optimizer, 
    build_bsam_optimizer, 
    build_bsam_svi_swa_optimizer,
    compute_posterior_predictive
)


num_workers = 2

def get_optimizer(args, ndata, modelapply):
    wdecay = args.priorprec / float(ndata)
    def nllloss(param, netstate, minibatch, is_training):
        logits, newstate = modelapply(param, netstate, None, minibatch[0], is_training)
        loss = nll_categorical(logits, minibatch[1])
        return loss, newstate
    
    if args.optim == 'sgd':
        optinit, optstep = build_sgd_optimizer(
            jax.value_and_grad(nllloss, has_aux=True),
            learningrate=args.alpha,
            momentum=args.beta1,
            wdecay=wdecay
        )
        
    elif args.optim == 'sam':
        optinit, optstep = build_sam_optimizer(
            jax.value_and_grad(nllloss, has_aux=True),
            learningrate=args.alpha,
            momentum=args.beta1,
            wdecay=wdecay,
            rho=args.rho,
            msharpness=args.batchsplit
        )
        
    elif args.optim == 'bsam':
        optinit, optstep = build_bsam_optimizer(
            jax.value_and_grad(nllloss, has_aux=True),
            learningrate=args.alpha,
            beta1=args.beta1,
            beta2=args.beta2,
            wdecay=wdecay,
            rho=args.rho,
            msharpness=args.batchsplit,
            Ndata=ndata,
            s_init=args.custominit,
            damping=args.damping
        )
    
    elif args.optim == 'bsam_svi_swa':
        optinit, optstep = build_bsam_svi_swa_optimizer(
            jax.value_and_grad(nllloss, has_aux=True),
            learningrate=args.alpha,
            beta1=args.beta1,
            beta2=args.beta2,
            wdecay=wdecay,
            rho=args.rho,
            msharpness=args.batchsplit,
            Ndata=ndata,
            s_init=args.custominit,
            damping=args.damping,
            swa_start=args.swa_start,
            swa_freq=args.swa_freq,
            temperature=args.temperature
        )
        
    else:
        print(f'Optimizer {args.optim} not implemented.')
        sys.exit()
    
    return optinit, optstep


def test_with_svi_swa(trainstate, modelapply, testloader, nclasses, num_mc_samples=3):
    correct = 0
    total = 0
    
    if trainstate.optstate['swa_n'] > 0:
        posterior_mean, posterior_var = compute_posterior_predictive(
            trainstate.optstate, num_mc_samples
        )
        
        rngkey = jax.random.PRNGKey(42)
        
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if batch_idx >= 20:
                break
                
            dat = jnp.array(inputs.numpy().transpose(0, 2, 3, 1))
            tgt = jax.nn.one_hot(targets.numpy(), nclasses)
            
            # Monte Carlo integration
            mc_predictions = []
            for i in range(num_mc_samples):
                noise, rngkey = normal_like_tree(posterior_mean, rngkey)
                theta_sample = jax.tree.map(
                    lambda mu, var, n: mu + jnp.sqrt(var) * n,
                    posterior_mean, posterior_var, noise
                )
                
                logits, _ = modelapply(
                    theta_sample, trainstate.netstate, None, dat, is_training=False
                )
                mc_predictions.append(jax.nn.softmax(logits, axis=1))
            
            # Bayesian model averaging
            avg_probs = jnp.mean(jnp.stack(mc_predictions), axis=0)
            preds = jnp.argmax(avg_probs, axis=1)
            correct += jnp.sum(preds == jnp.argmax(tgt, axis=1))
            total += dat.shape[0]
    else:
        theta = trainstate.optstate['w']
        
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if batch_idx >= 20:
                break
                
            dat = jnp.array(inputs.numpy().transpose(0, 2, 3, 1))
            tgt = jax.nn.one_hot(targets.numpy(), nclasses)
            
            logits, _ = modelapply(theta, trainstate.netstate, None, dat, is_training=False)
            correct += jnp.sum(logits.argmax(axis=1) == tgt.argmax(axis=1))
            total += dat.shape[0]
    
    return float(correct) / float(total)


def main():
    parser = argparse.ArgumentParser()  
    parser.add_argument('--optim', type=str, default='bsam_svi_swa', choices=['sgd', 'sam', 'bsam', 'bsam_svi_swa'])
    parser.add_argument('--alpha', type=float, default=0.5, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='momentum for gradient')
    parser.add_argument('--beta2', type=float, default=0.999, help='momentum for variance')
    parser.add_argument('--rho', type=float, default=0.05, help='parameter for SAM optimizers')
    parser.add_argument('--priorprec', type=float, default=40.0, help='prior precision')
    parser.add_argument('--batchsize', type=int, default=200)
    parser.add_argument('--testbatchsize', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=180)
    parser.add_argument('--warmup', type=int, default=5, help='linear learning-rate warmup')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--batchsplit', type=int, default=8, help='independent perturbations on subbatches?')
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--datasetfolder', type=str, default='datasets')
    parser.add_argument('--resultsfolder', type=str, default='results')
    parser.add_argument('--custominit', type=float, default=1.0, help='special initialization value for variance')
    parser.add_argument('--damping', type=float, default=0.1, help='damping to stabilize the method')
    parser.add_argument('--dafactor', type=float, default=4.0, help='multiplicative factor to adjust size of dataset')
    parser.add_argument('--swa_start', type=float, default=0.75, help='fraction of training to start SWA')
    parser.add_argument('--swa_freq', type=int, default=5, help='steps between SWA updates')
    parser.add_argument('--temperature', type=float, default=0.5, help='temperature for adaptive LR')
    parser.add_argument('--noaugment', dest='augment', action='store_false', help='no data augmentation')
    parser.set_defaults(augment=True)
    
    args = parser.parse_args()
    
    idx = 0
    while True:
        outpath = f"{args.resultsfolder}/{args.dataset}_{args.model}/{args.optim}/run_{idx}"
        if not os.path.exists(outpath):
            break
        idx += 1
    os.makedirs(outpath)
    
    print(f"Training with SVI-SWA optimizer")
    print(f"Output directory: {outpath}")
    print("\nHyperparameters:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    
    rngkey = jax.random.PRNGKey(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Load dataset
    trainset, testset, trainloader, testloader = dataloader(args.dataset)(
        args.batchsize, args.testbatchsize, args.datasetfolder,
        args.augment, num_workers
    )
    
    ndata = len(trainset)
    nclasses = len(trainset.classes)
    ndata *= args.dafactor
    
    print(f"\nDataset: {args.dataset}")
    print(f"Training samples: {len(trainset)}")
    print(f"Test samples: {len(testset)}")
    print(f"Classes: {nclasses}")
    
    modelapply, modelinit = get_model(args.model.lower(), nclasses)
    
    datapoint = next(iter(trainloader))[0].numpy().transpose(0, 2, 3, 1)
    params, netstate = modelinit(rngkey, datapoint, True)
    
    numparams = sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(params))
    print(f"\nModel: {args.model} ({numparams:,} parameters)")
    
    optinit, optstep = get_optimizer(args, ndata, modelapply)
    trainstate = optinit(params, netstate, rngkey)
    
    print("\nPre-compiling...")
    dummy_inputs, dummy_targets = next(iter(trainloader))
    X_dummy = jnp.array(dummy_inputs.numpy().transpose(0, 2, 3, 1))
    y_dummy = jax.nn.one_hot(dummy_targets.numpy(), nclasses)
    
    if args.optim == 'bsam_svi_swa':
        trainstate, _ = optstep(trainstate, (X_dummy, y_dummy), 1.0, 0)
    else:
        trainstate, _ = optstep(trainstate, (X_dummy, y_dummy), 1.0)
    print("Pre-compilation complete. Starting training...")
    
    # Training
    step_counter = 0
    best_acc = 0.0
    
    for epoch in trange(args.epochs + 1, desc="Training"):
        if epoch < args.warmup:
            lrfactor = jnp.linspace(0.0, 1.0, args.warmup + 1)[epoch + 1]
        else:
            step_t = float(epoch - args.warmup) / max(1, args.epochs - args.warmup)
            lrfactor = 0.5 * (1.0 + jnp.cos(jnp.pi * step_t))
        
        # Train for one epoch
        epoch_losses = []
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            X = jnp.array(inputs.numpy().transpose(0, 2, 3, 1))
            y = jax.nn.one_hot(targets.numpy(), nclasses)
            
            if args.optim == 'bsam_svi_swa':
                trainstate, loss = optstep(trainstate, (X, y), lrfactor, step_counter)
            else:
                trainstate, loss = optstep(trainstate, (X, y), lrfactor)
            
            epoch_losses.append(float(loss))
            step_counter += 1
        
        avg_loss = np.mean(epoch_losses)
        
        # Evaluate
        if epoch % 10 == 0 or epoch == args.epochs:
            if args.optim == 'bsam_svi_swa':
                acc = test_with_svi_swa(trainstate, modelapply, testloader, nclasses) * 100.0
            else:
                correct = 0
                total = 0
                for batch_idx, (inputs, targets) in enumerate(testloader):
                    if batch_idx >= 20:
                        break
                    dat = jnp.array(inputs.numpy().transpose(0, 2, 3, 1))
                    tgt = jax.nn.one_hot(targets.numpy(), nclasses)
                    theta = trainstate.optstate['w']
                    logits, _ = modelapply(theta, trainstate.netstate, None, dat, is_training=False)
                    correct += jnp.sum(logits.argmax(axis=1) == tgt.argmax(axis=1))
                    total += dat.shape[0]
                acc = 100.0 * float(correct) / float(total)
            
            if acc > best_acc:
                best_acc = acc
                
            swa_info = ""
            if args.optim == 'bsam_svi_swa' and trainstate.optstate['swa_n'] > 0:
                swa_info = f", SWA: {trainstate.optstate['swa_n']} snapshots"
            
            tprint(f"Epoch {epoch:3d}/{args.epochs}: "
                  f"Trainloss = {avg_loss:.4f}, "
                  f"Acc = {acc:.2f}% (best: {best_acc:.2f}%)"
                  f"{swa_info}")
            
            with open(os.path.join(outpath, 'trainstate.pickle'), 'wb') as f:
                pickle.dump(trainstate, f)
                pickle.dump(args, f)
    
    print(f"\nTraining completed. Best accuracy: {best_acc:.2f}%")
    print(f"Results are saved in {outpath}")


if __name__ == '__main__':
    main()