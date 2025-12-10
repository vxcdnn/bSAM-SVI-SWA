import jax
import jax.numpy as jnp
import numpy as np
import copy 
from util import normal_like_tree
from typing import NamedTuple

class TrainState(NamedTuple):
    """
    collects the all the state required for neural network training
    """
    optstate: dict
    netstate: None
    rngkey: None

def build_sgd_optimizer(lossgrad,
                        learningrate : float,
                        momentum : float,
                        wdecay : float): 

    def init(weightinit, netstate, rngkey): 
        optstate = dict()
        optstate['w'] = copy.deepcopy(weightinit)
        optstate['gm'] = jax.tree.map(lambda p : jnp.zeros(shape=p.shape), weightinit)  
        optstate['alpha'] = learningrate 

        return TrainState(optstate = optstate,
                          netstate = netstate,
                          rngkey = rngkey)

    def step(trainstate, minibatch, lrfactor):
        optstate = trainstate.optstate

        (loss, netstate), grad = lossgrad(optstate['w'], trainstate.netstate, minibatch, is_training=True) 

        # momentum
        optstate['gm'] = jax.tree.map(
            lambda gm, g, w: momentum * gm + g + wdecay * w, optstate['gm'], grad, optstate['w'])

        # weight update 
        optstate['w'] = jax.tree.map(lambda p, gm: p - learningrate * lrfactor * gm, optstate['w'], optstate['gm'])
    
        newtrainstate = trainstate._replace(
            optstate = optstate,
            netstate = netstate)

        return newtrainstate, loss

    return init, step

def build_sam_optimizer(lossgrad,
                        learningrate : float,
                        momentum : float,
                        wdecay : float,
                        rho : float,
                        msharpness : int): 

    def init(weightinit, netstate, rngkey): 
        optstate = dict()
        optstate['w'] = copy.deepcopy(weightinit)
        optstate['gm'] = jax.tree.map(lambda p : jnp.zeros(shape=p.shape), weightinit)  
        optstate['alpha'] = learningrate 

        return TrainState(optstate = optstate,
                          netstate = netstate,
                          rngkey = rngkey)
    
    def _sam_gradient(trainstate, X_subbatch, y_subbatch):
        (_, netstate), grad = lossgrad(trainstate.optstate['w'], trainstate.netstate, (X_subbatch, y_subbatch), is_training = True) 
        grad_norm = jnp.sqrt(sum([jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(grad)]))
        perturbed_params = jax.tree.map(lambda p, g: p + rho * g / grad_norm, trainstate.optstate['w'], grad)
        (loss, netstate), perturbed_grad= lossgrad(perturbed_params, netstate, (X_subbatch, y_subbatch), is_training = True)

        return perturbed_grad, netstate, loss        

    def step(trainstate, minibatch, lrfactor):
        optstate = trainstate.optstate

        # split batch to simulate m-sharpness on one GPU
        X_batch = minibatch[0].reshape(msharpness, -1, *minibatch[0].shape[1:])
        y_batch = minibatch[1].reshape(msharpness, -1, *minibatch[1].shape[1:]) 
        grad, netstate, loss = jax.vmap(_sam_gradient, in_axes=(None, 0, 0))(trainstate, X_batch, y_batch)
        grad = jax.tree.map(lambda g : jnp.mean(g, axis=0), grad)
        netstate = jax.tree.map(lambda p : p[0], netstate) 
        loss = jnp.mean(loss)

        # momentum
        optstate['gm'] = jax.tree.map(
            lambda gm, g, w: momentum * gm + g + wdecay * w, optstate['gm'], grad, optstate['w'])

        # weight update 
        optstate['w'] = jax.tree.map(lambda p, gm: p - learningrate * lrfactor * gm, optstate['w'], optstate['gm'])
    
        newtrainstate = trainstate._replace(
            optstate = optstate,
            netstate = netstate)

        return newtrainstate, loss

    return init, step


def build_bsam_optimizer(lossgrad,
                         learningrate : float,
                         beta1 : float,
                         beta2 : float,
                         wdecay : float,
                         rho : float,
                         msharpness : int,
                         Ndata : int, 
                         s_init : float, 
                         damping : float): 

    def init(weightinit, netstate, rngkey): 
        optstate = dict()
        optstate['w'] = copy.deepcopy(weightinit)
        optstate['gm'] = jax.tree.map(lambda p : jnp.zeros(shape=p.shape), weightinit)  
        optstate['alpha'] = learningrate 
        optstate['s'] = jax.tree.map(lambda p : s_init * jnp.ones(shape=p.shape), weightinit)

        return TrainState(optstate = optstate,
                          netstate = netstate,
                          rngkey = rngkey)
    
    def _bsam_gradient(trainstate, X_subbatch, y_subbatch, rngkey):
        optstate = trainstate.optstate

        # noisy sample
        noise, _ = normal_like_tree(optstate['w'], rngkey)
        noisy_param = jax.tree.map(lambda n, mu, s: mu + \
            jnp.sqrt(1.0 / (Ndata * s)) * n, noise, optstate['w'], optstate['s'])

        # gradient at noisy sample 
        (_, netstate), grad = lossgrad(noisy_param, trainstate.netstate, (X_subbatch, y_subbatch), is_training = True)

        perturbed_params = jax.tree.map(lambda p, g, s: p + rho * g / s, optstate['w'], grad, optstate['s'])
        (loss, netstate), perturbed_grad = lossgrad(perturbed_params, netstate, (X_subbatch, y_subbatch), is_training = True)

        gs = jax.tree.map(lambda g, s: jnp.sqrt(s * (g ** 2.0)), grad, optstate['s'])

        return gs, perturbed_grad, netstate, loss     

    def step(trainstate, minibatch, lrfactor):
        optstate = trainstate.optstate
        rngkey = trainstate.rngkey

        # split batch to simulate m-sharpness on one GPU
        rngkeys = jax.random.split(rngkey, msharpness + 1)
        X_batch = minibatch[0].reshape(msharpness, -1, *minibatch[0].shape[1:])
        y_batch = minibatch[1].reshape(msharpness, -1, *minibatch[1].shape[1:]) 
        gs, grad, netstate, loss = jax.vmap(_bsam_gradient, in_axes=(None, 0, 0, 0))(trainstate, X_batch, y_batch, rngkeys[0:msharpness])

        gs = jax.tree.map(lambda g : jnp.mean(g, axis=0), gs)
        grad = jax.tree.map(lambda g : jnp.mean(g, axis=0), grad)
        netstate = jax.tree.map(lambda p : p[0], netstate) 
        loss = jnp.mean(loss)

        # momentum
        optstate['gm'] = jax.tree.map(
            lambda gm, g, w: beta1 * gm + (1 - beta1) * (g + wdecay * w), optstate['gm'], grad, optstate['w'])

        # weight update 
        optstate['w'] = jax.tree.map(lambda p, gm, s: p - learningrate * lrfactor * gm / s, optstate['w'], optstate['gm'], optstate['s'])
    
        # update precision 
        optstate['s'] = jax.tree.map(lambda s, gs: beta2 * s + (1 - beta2) * (gs + damping + wdecay), 
                                     optstate['s'], gs)

        newtrainstate = trainstate._replace(
            optstate = optstate,
            netstate = netstate,
            rngkey = rngkeys[-1])

        return newtrainstate, loss

    return init, step



def build_bsam_svi_swa_optimizer(lossgrad,
                                 learningrate: float,
                                 beta1: float,
                                 beta2: float,
                                 wdecay: float,
                                 rho: float,
                                 msharpness: int,
                                 Ndata: int,
                                 s_init: float,
                                 damping: float,
                                 swa_start: float = 0.75,
                                 swa_freq: int = 5,
                                 temperature: float = 1.0):
    
    def init(weightinit, netstate, rngkey): 
        optstate = dict()
        optstate['w'] = copy.deepcopy(weightinit)
        optstate['gm'] = jax.tree.map(lambda p: jnp.zeros(shape=p.shape), weightinit)  # Gradient momentum
        optstate['s'] = jax.tree.map(lambda p: s_init * jnp.ones(shape=p.shape), weightinit)  # Precision
        
        optstate['swa_w_mean'] = copy.deepcopy(weightinit)  # SWA mean
        optstate['swa_w_m2'] = jax.tree.map(lambda p: jnp.zeros_like(p), weightinit)  # Sum of squared diffs
        optstate['swa_n'] = jnp.array(0, dtype=jnp.int32)  # SWA counter
        
        optstate['step'] = jnp.array(0, dtype=jnp.int32)
        optstate['ema_grad_norm'] = jnp.array(1.0)
        
        return TrainState(optstate=optstate,
                          netstate=netstate,
                          rngkey=rngkey)
    
    @jax.jit
    def _bsam_gradient(trainstate, minibatch, rngkey):
        optstate = trainstate.optstate
        
        noise, _ = normal_like_tree(optstate['w'], rngkey)
        noisy_param = jax.tree.map(
            lambda n, mu, s: mu + jnp.sqrt(1.0 / (Ndata * s)) * n,
            noise, optstate['w'], optstate['s']
        )
        
        # gradient at noisy sample
        (_, netstate), grad = lossgrad(
            noisy_param, trainstate.netstate, minibatch, is_training=True
        )
        
        # SAM perturbation
        perturbed_params = jax.tree.map(
            lambda p, g, s: p + rho * g / (jnp.sqrt(s) + 1e-8),
            optstate['w'], grad, optstate['s']
        )
        (loss, netstate), perturbed_grad = lossgrad(
            perturbed_params, netstate, minibatch, is_training=True
        )
        
        gs = jax.tree.map(lambda g, s: jnp.sqrt(s) * (g ** 2), grad, optstate['s'])
        
        return gs, perturbed_grad, netstate, loss
    
    @jax.jit
    def step(trainstate, minibatch, lrfactor, step_counter):
        optstate = trainstate.optstate
        rngkey = trainstate.rngkey
        
        # split batch to simulate m-sharpness on one GPU
        rngkeys = jax.random.split(rngkey, msharpness + 1)
        X_batch = minibatch[0].reshape(msharpness, -1, *minibatch[0].shape[1:])
        y_batch = minibatch[1].reshape(msharpness, -1, *minibatch[1].shape[1:])
        
        # gradient computation
        gs, grad, netstate, loss = jax.vmap(
            lambda X, y, key: _bsam_gradient(
                trainstate._replace(rngkey=key), (X, y), key
            ),
            in_axes=(0, 0, 0)
        )(X_batch, y_batch, rngkeys[:msharpness])
        
        # average over sub-batches
        gs = jax.tree.map(lambda g: jnp.mean(g, axis=0), gs)
        grad = jax.tree.map(lambda g: jnp.mean(g, axis=0), grad)
        netstate = jax.tree.map(lambda p: p[0], netstate)
        loss = jnp.mean(loss)
        
        # momentum
        optstate['gm'] = jax.tree.map(
            lambda gm, g, w: beta1 * gm + (1 - beta1) * (g + wdecay * w),
            optstate['gm'], grad, optstate['w']
        )
        
        # adaptive learning rate based on gradient statistics
        grad_norm = jnp.sqrt(sum([
            jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(grad)
        ]))
        ema_decay = 0.99
        optstate['ema_grad_norm'] = (
            ema_decay * optstate['ema_grad_norm'] + 
            (1 - ema_decay) * grad_norm
        )
        
        adaptive_lr = learningrate * temperature / (1.0 + optstate['ema_grad_norm'])
        
        # weight update
        optstate['w'] = jax.tree.map(
            lambda w, gm, s: w - adaptive_lr * lrfactor * gm / (jnp.sqrt(s) + 1e-8),
            optstate['w'], optstate['gm'], optstate['s']
        )
        
        # update precision 
        optstate['s'] = jax.tree.map(
            lambda s, gs_val: beta2 * s + (1 - beta2) * (gs_val + damping + wdecay),
            optstate['s'], gs
        )
        
        # update SWA statistics
        step_counter_jax = jnp.array(step_counter, dtype=jnp.int32)
        swa_start_step = jnp.array(int(swa_start * 100000), dtype=jnp.int32)
        
        def update_swa(optstate):
            n = optstate['swa_n'] + 1
            
            # Online mean update using Welford's algorithm
            delta = jax.tree.map(
                lambda new, old: new - old,
                optstate['w'], optstate['swa_w_mean']
            )
            
            # update mean
            new_mean = jax.tree.map(
                lambda old, d: old + d / n,
                optstate['swa_w_mean'], delta
            )
            
            delta2 = jax.tree.map(
                lambda new, mean: new - mean,
                optstate['w'], new_mean
            )
            new_m2 = jax.tree.map(
                lambda m2, d1, d2: m2 + d1 * d2,
                optstate['swa_w_m2'], delta, delta2
            )
            
            return {
                **optstate,
                'swa_w_mean': new_mean,
                'swa_w_m2': new_m2,
                'swa_n': n
            }
        
        def no_update(optstate):
            """No SWA update"""
            return optstate
        
        # conditionally update SWA
        should_update = (step_counter_jax > swa_start_step) & (
            step_counter_jax % swa_freq == 0
        )
        optstate = jax.lax.cond(
            should_update,
            update_swa,
            no_update,
            optstate
        )
        
        # Update step counter
        optstate['step'] = step_counter_jax
        
        newtrainstate = trainstate._replace(
            optstate=optstate,
            netstate=netstate,
            rngkey=rngkeys[-1]
        )
        
        return newtrainstate, loss
    
    return init, step


def compute_posterior_predictive(optstate, num_samples=5):
    if optstate['swa_n'] == 0:
        # fall back to current variational distribution
        mean = optstate['w']
        # approximate variance from precision
        variance = jax.tree.map(lambda s: 1.0 / (optstate['swa_n'] * s + 1e-8), optstate['s'])
        return mean, variance
    
    mean = optstate['swa_w_mean']
    
    # compute empirical variance from M2
    n = optstate['swa_n']
    variance = jax.tree.map(
        lambda m2: m2 / jnp.maximum(n - 1, 1) + 1e-8,
        optstate['swa_w_m2']
    )
    
    return mean, variance