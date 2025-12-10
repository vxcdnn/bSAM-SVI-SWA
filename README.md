# bSAM-SVI-SWA

This is a repository of bSAM upgrade learning project. To train and test our method you should do the following steps:

1) First you need to create Python 3.10 venv on Linux or WSL2 (Windows is not natively supported by JAX):

python3.10 -m venv bsam-env
source bsam-env/bin/activate

2) Then install required packages and CUDA 12.8 (if not installed) in the following order:
pip install typing-extensions==4.12.2
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r requirements.txt

3) For training with ResNet18 on CIFAR10 dataset, type:

python train.py --alpha 0.5 --beta1 0.9 --beta2 0.999 --priorprec 40 --rho 0.05 --batchsplit 8 --optim bsam_swi_swa --dataset cifar10 --dafactor 4 --batchsize 128

4) For testing:

python test.py --resultsfolder results/cifar10_resnet18/bsam/run_# --testmc 32


