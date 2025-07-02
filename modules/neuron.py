from typing import Callable

import torch
from spikingjelly.clock_driven.neuron import LIFNode as LIFNode_sj
from spikingjelly.clock_driven.neuron import ParametricLIFNode as PLIFNode_sj
from torch import nn

from modules.surrogate import Rectangle


# multistep torch version
class ILIFSpike(nn.Module):
    def __init__(self, tau: float):
        super(ILIFSpike, self).__init__()
        # the symbol is corresponding to the paper
        # self.spike_func = surrogate_function
        self.spike_func = Rectangle()

        self.v_th = 1.
        self.gamma = 1 - 1. / tau

    def forward(self, x_seq):
        # x_seq.shape should be [T, N, *]
        spike_sequence = []
        voltage = 0
        memory = 0
        T = x_seq.shape[0]
        for t in range(T):
            voltage = self.gamma * voltage + x_seq[t, ...]
            spike = self.spike_func(voltage - self.v_th)
            spike_sequence.append(spike)
            memory = memory * torch.sigmoid_((1. - self.gamma) * voltage) + spike
            voltage = voltage - spike * (self.v_th + torch.sigmoid_(memory))
        # self.pre_spike_mem = torch.stack(_mem)
        return torch.stack(spike_sequence, dim=0)


# spikingjelly single step version
class InhibitoryLIFNeuron(LIFNode_sj):
    def __init__(self, tau: float = 2.,
                 decay_input: bool = False, v_threshold: float = 1.,
                 v_reset: float = None, surrogate_function: Callable = Rectangle(),
                 detach_reset: bool = False, cupy_fp32_inference=False, **kwargs):
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, cupy_fp32_inference)
        self.register_memory('inhibitory_memory', 0.)  # Inhibitory memory
        self.register_memory('prev_input', 0.)
        self.register_memory('prev_spike', 0.)
        self.register_memory('inhibition', 0.)

    def forward(self, x: torch.Tensor):
        if isinstance(self.prev_input, float):
            self.prev_input = torch.zeros_like(x)
            self.prev_spike = torch.zeros_like(x)
            self.inhibition = torch.zeros_like(x)
        current_input = x
        self.inhibition = 0.03*(self.inhibition + self.prev_input * self.prev_spike)
        self.neuronal_charge(x-torch.clamp(self.inhibition,min=0))  # LIF charging
        self.prev_input = current_input
        spike = self.neuronal_fire()  # LIF fire
        self.prev_spike = spike
        self.neuronal_reset(spike)  # LIF reset
        self.inhibitory_memory = 1*(self.inhibitory_memory + spike * self.v)
        self.v = self.v - spike * torch.sigmoid(self.inhibitory_memory)  # Reset
        return spike


    def neuronal_charge(self, x: torch.Tensor):
        self._charging_v(x)

    def neuronal_reset(self, spike: torch.Tensor):
        self._reset(spike)

    def _charging_v(self, x: torch.Tensor):
        if self.decay_input:
            x = x / self.tau

        if self.v_reset is None or self.v_reset == 0:
            if type(self.v) is float:
                self.v = x
            else:
                self.v = self.v * (1 - 1. / self.tau) + x
        else:
            if type(self.v) is float:
                self.v = self.v_reset * (1 - 1. / self.tau) + self.v_reset / self.tau + x
            else:
                self.v = self.v * (1 - 1. / self.tau) + self.v_reset / self.tau + x

    def _reset(self, spike):
        if self.v_reset is None:
            # soft reset
            self.v = self.v - spike * self.v_threshold
        else:
            # hard reset
            self.v = (1. - spike) * self.v + spike * self.v_reset


# spikingjelly multiple step version
class MultiStepILIFNeuron(InhibitoryLIFNeuron):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
                 v_reset: float = None, surrogate_function: Callable = Rectangle(),
                 detach_reset: bool = False, cupy_fp32_inference=False, **kwargs):
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, cupy_fp32_inference)

    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() > 1
        # x_seq.shape = [T, *]
        spike_seq = []
        self.v_seq = []
        for t in range(x_seq.shape[0]):
            spike_seq.append(super().forward(x_seq[t]).unsqueeze(0))
            self.v_seq.append(self.v.unsqueeze(0))
        spike_seq = torch.cat(spike_seq, 0)
        self.v_seq = torch.cat(self.v_seq, 0)
        return spike_seq


class ReLU(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return torch.relu(x)


class BPTTNeuron(LIFNode_sj):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
                 v_reset: float = None, surrogate_function: Callable = Rectangle(),
                 detach_reset: bool = False, cupy_fp32_inference=False, **kwargs):
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, cupy_fp32_inference)


class PLIFNeuron(PLIFNode_sj):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
                 v_reset: float = None, surrogate_function: Callable = None,
                 detach_reset: bool = False, cupy_fp32_inference=False, **kwargs):
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset)


if __name__ == '__main__':
    T = 8
    x_input = torch.rand((T, 3, 32, 32)) * 1.2
    ilif = InhibitoryLIFNeuron()
    ilif_m = MultiStepILIFNeuron()

    s_list = []
    for t in range(T):
        s = ilif(x_input[t])
        s_list.append(s)

    s_list = torch.stack(s_list, dim=0)
    s_output = ilif_m(x_input)

    print(s_list.mean())
    print(s_output.mean())
    assert torch.sum(s_output - torch.Tensor(s_list)) == 0
