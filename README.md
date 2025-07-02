# ILIF

The Leaky Integrate-and-Fire (LIF) model faces a dilemma in configuring the surrogate gradient support width, balancing between overactivation and gradient vanishing. This work proposes the Temporal Inhibitory Leaky Integrate-and-Fire (ILIF) Neuron, which incorporates inhibitory units connected across adjacent time steps to mitigate overactivation and ensure smooth gradient propagation.


## Dependencies
- Python 3
- PyTorch, torchvision
- spikingjelly 0.0.0.0.12
- Python packages: `pip install tqdm progress torchtoolbox thop`


## Training
We use one single RTX4090D GPU for running all the experiments. Multi-GPU training is not supported in the current codes.

For training, please run the following example codes:

    # CIFAR10
    python train.py -data_dir ./data -dataset cifar10 -model spiking_resnet18 -T_max 200 -epochs 200 -weight_decay 5e-5 -neuron ILIF
    
    # CIFAR100
    python train.py -data_dir ./data -dataset cifar100 -model spiking_resnet18 -T_max 200 -epochs 200 -neuron ILIF
       
    # DVSCIFAR10
    python train.py -data_dir ./data -dataset DVSCIFAR10 -T 10 -drop_rate 0.3 -model spiking_vgg11_bn -lr=0.05 -mse_n_reg -neuron ILIF
    
    # DVSGesture
    python train.py -data_dir ./data -dataset dvsgesture -model spiking_vgg11_bn -T 20 -b 16 -drop_rate 0.4 -neuron ILIF

If you change the neuron, you can directly switch to ``LIF`` or ``PLIF`` by modifying the hyperparameters after ``-neuron``.

## Inference
For inference, please run the following example codes:

    # CIFAR10
    python inference.py -data_dir ./data -dataset cifar10 -model spiking_resnet18 -neuron LIF -name without_auto_aug -resume ./logs/checkpoint_max.pth
    
    python inference.py -data_dir ./data -dataset cifar10 -model spiking_resnet18 -neuron ILIF -name without_auto_aug -resume ./logs/checkpoint_max.pth
    
    # CIFAR100
    python inference.py -data_dir ./data -dataset cifar100 -model spiking_resnet18 -neuron LIF -name without_auto_aug -resume ./logs/checkpoint_max.pth
    
    python inference.py -data_dir ./data -dataset cifar100 -model spiking_resnet18 -neuron ILIF -name without_auto_aug -resume ./logs/checkpoint_max.pth
    
    # DVSCIFAR10
    python inference.py -data_dir ./data -dataset DVSCIFAR10 -model spiking_vgg11_bn -T 10 -neuron LIF -name without_auto_aug -resume ./logs/checkpoint_max.pth
    
    python inference.py -data_dir ./data -dataset DVSCIFAR10 -model spiking_vgg11_bn -T 10 -neuron ILIF -name without_auto_aug -resume ./logs/checkpoint_max.pth
    
    # DVSGesture
    python inference.py -data_dir ./data -dataset dvsgesture -model spiking_vgg11_bn -T 20 -b 16 -neuron LIF -name without_auto_aug -resume ./logs/checkpoint_max.pth
    
    python inference.py -data_dir ./data -dataset dvsgesture -model spiking_vgg11_bn -T 20 -b 16 -neuron ILIF -name without_auto_aug -resume ./logs/checkpoint_max.pth

## Acknowlegement
The code for data preprocessing and neuron models is based on the [spikingjelly](https://github.com/fangwei123456/spikingjelly) repo. Our implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.

[Codebase1](https://github.com/qymeng94/SLTT) [Codebase2](https://github.com/HuuYuLong/Complementary-LIF)
