# bSAM-SVI-SWA

This is the repository of bSAM upgrade learning project. To train and test our method, follow these steps:

1) **Set up the environment:** First you need to create Python 3.10 venv on Linux or WSL2 (Windows is not natively supported by JAX):

python3.10 -m venv bsam-env

source bsam-env/bin/activate

2) **Install dependencies:** Then, install the required packages in the following order:

pip install typing-extensions==4.12.2

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

pip install -r requirements.txt

*Note: Ensure CUDA 12.8 is installed if not present.*

3) **Training:** To train with ResNet-18 on CIFAR10 dataset, run:

python train.py --alpha 0.5 --beta1 0.9 --beta2 0.999 --priorprec 40 --rho 0.05 --batchsplit 8 --optim bsam_swi_swa --dataset cifar10 --dafactor 4 --batchsize 128

4) **Testing:** To test a trained model, use:

python test.py --resultsfolder results/cifar10_resnet18/bsam/run_0 --testmc 32

Additionally, a PyTorch implementation of the original SAM optimizer is provided in the following files: Example_of_use_torch_optim.ipynb and torch_optim.py.
