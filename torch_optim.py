import numpy as np
import torch
from typing import Callable, Optional
from tqdm import trange

from data import dataloader

class SamOptimizer(torch.optim.Optimizer):
    def __init__(
        self, params,
        bayesian: bool,
        lr : float,
        beta1 : float,
        beta2 : float,
        wdecay : float,
        rho : float,
        Ndata : Optional[int] = None, 
        s_init : Optional[float] = 0, 
        damping : Optional[float] = None,
    ):
        super(SamOptimizer, self).__init__(params, {})
        self.bayesian = bayesian
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.wdecay = wdecay
        self.rho = rho
        self.Ndata = Ndata
        self.s_init = s_init
        self.damping = damping
        self.state = dict()
        for group in self.param_groups:
            for p in group['params']:
                self.state[p] = dict(
                    gm=torch.zeros_like(p.data),
                    s=torch.full_like(p.data, s_init),
                )
    
    def step(self, loss_function: Callable):
        params = []
        for group in self.param_groups:
            params += [(p, self.state[p]) for p in group['params']]

        for p, st in params:
            st["w"] = p.data
            if self.bayesian:
                p.data = st["w"] + torch.normal(0, 1 / (self.Ndata * st["s"]))
        loss_function()

        if self.bayesian:
            for p, st in params:
                st["g"] = p.grad.data
        else:
            norm = torch.sqrt(sum([(p.grad.data ** 2).sum() for p, st in params]))
        for p, st in params:
            if self.bayesian:
                norm = st["s"]
            p.data = st["w"] + self.rho * p.grad.data / norm
        loss = loss_function()

        for p, st in params:
            g = p.grad.data
            st["gm"] += (1 - self.beta1) * (g + self.wdecay * p.data - st["gm"])
            if self.bayesian:
                s_ = torch.sqrt(st["s"]) * torch.abs(st["g"]) + self.damping + self.wdecay
            else:
                s_ = g + self.wdecay * st["w"]
            st["s"] += (1 - self.beta2) * (s_ - st["s"])
            m = st["s"] if self.bayesian else torch.sqrt(st["s"]) + self.damping
            p.data = st["w"] - self.lr * st["gm"] / m
        return loss.detach()


def nll_categorical(logits, labels):
    """ multiclass classification negative log-likelihood """
    loss = -torch.sum(logits * labels, axis=1) + torch.logsumexp(logits, axis=1)
    return torch.mean(loss, axis=0)

def main(model, device, epochs, warmup, dataset, batchsize, testbatchsize, datasetfolder, augment, **kwargs):
    num_workers = 4
    trainset, testset, trainloader, testloader = dataloader(dataset)(
        batchsize,
        testbatchsize,
        datasetfolder,
        augment,
        num_workers,
    )

    ndata = len(trainset)
    ntestdata = len(testset)
    nclasses = len(trainset.classes)
    
    if "priorprec" in kwargs:
        kwargs["wdecay"] = kwargs.pop("priorprec") / ndata
    optimizer = SamOptimizer(model.parameters(), Ndata=ndata, **kwargs)

    def get_loss():
        loss = nll_categorical(model(inputs.to(device)), y.to(device))
        loss.backward()
        return loss
        
    losses = []
    for epoch in trange(
        epochs + 1, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}", smoothing=1.0
    ):
        if epoch < warmup:
            lrfactor = torch.linspace(0.0, 1.0, warmup + 1)[epoch + 1]
        else:
            step_t = float(epoch - warmup) / float(epochs + 1 - warmup)
            lrfactor = 0.5 * (1.0 + np.cos(np.pi * step_t))

        optimizer.lr = kwargs["lr"] * lrfactor
        loss = []
        for inputs, targets in trainloader:
            y = torch.nn.functional.one_hot(targets, nclasses)
            optimizer.zero_grad()
            loss.append(optimizer.step(get_loss))
        loss = torch.tensor(loss).mean()
        losses.append(loss)
        
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                logits = model(inputs.to(device))
                correct += sum(logits.argmax(axis=1) == targets.to(device)).sum()
                total += logits.shape[0]
        acc = correct / total
        print(f"[{epoch:3d}/{epochs}] Trainloss (at samples): {loss:.3f} | Acc: {acc:.3f} ")
    return losses
