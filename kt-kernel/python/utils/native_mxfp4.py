"""Owning native MXFP4 storage. KT borrows its six CPU buffers without repacking."""
from contextlib import ExitStack
from dataclasses import dataclass
import ctypes
import json
import os
from pathlib import Path
import torch


@dataclass
class NativeMXFP4Weights:
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor
    gate_scale: torch.Tensor
    up_scale: torch.Tensor
    down_scale: torch.Tensor

    @property
    def tensors(self):
        return (self.gate, self.up, self.down, self.gate_scale, self.up_scale, self.down_scale)

    @property
    def nbytes(self):
        return sum(t.numel() for t in self.tensors)

    def validate(self, experts, hidden, intermediate):
        shapes = ((experts, intermediate, hidden // 2), ) * 2 + ((experts, hidden, intermediate // 2), ) + (
            (experts, intermediate, hidden // 32), ) * 2 + ((experts, hidden, intermediate // 32), )
        for tensor, shape in zip(self.tensors, shapes):
            if tensor.device.type != 'cpu' or tensor.dtype != torch.uint8 or not tensor.is_contiguous() or tuple(
                    tensor.shape) != shape:
                raise ValueError(
                    f'Native MXFP4 needs contiguous CPU uint8 {shape}; got {tensor.shape}/{tensor.dtype}/{tensor.device}'
                )
        if hidden % 256 or intermediate % 256:
            raise ValueError('KT native MXFP4 dimensions must be divisible by 256')
        return self

    @classmethod
    def from_checkpoint(cls, path, layer_idx, experts, hidden, intermediate, numa_node=0):
        from safetensors import safe_open
        root = Path(path)
        index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
        prefix = next((p for p in (f'layers.{layer_idx}.ffn.experts', f'model.layers.{layer_idx}.ffn.experts')
                       if f'{p}.0.w1.weight' in index), None)
        if prefix is None:
            raise ValueError(f'No native MXFP4 expert weights for layer {layer_idx}')
        # Bind this loading thread only. Explicit first-touch puts owned pages on the chosen node.
        previous = os.sched_getaffinity(0)
        text = Path(f'/sys/devices/system/node/node{numa_node}/cpulist').read_text().strip()
        cpus = set()
        for part in text.split(','):
            bounds = part.split('-')
            cpus.update(range(int(bounds[0]), int(bounds[-1]) + 1))
        cpus &= previous
        if not cpus: raise ValueError(f'NUMA node {numa_node} has no permitted CPUs')
        os.sched_setaffinity(0, cpus)
        try:
            shapes = ((experts, intermediate, hidden // 2), ) * 2 + ((experts, hidden, intermediate // 2), ) + (
                (experts, intermediate, hidden // 32), ) * 2 + ((experts, hidden, intermediate // 32), )
            buffers = [torch.empty(s, dtype=torch.uint8, device='cpu') for s in shapes]
            for tensor in buffers:
                ctypes.memset(tensor.data_ptr(), 0, tensor.numel())
            # Source maps are closed after load, leaving only the owned native buffers.
            with ExitStack() as stack:
                shards = {}
                for proj, wi, si in [('w1', 0, 3), ('w3', 1, 4), ('w2', 2, 5)]:
                    for eid in range(experts):
                        for suffix, target in [('weight', wi), ('scale', si)]:
                            key = f'{prefix}.{eid}.{proj}.{suffix}'
                            name = index[key]
                            if name not in shards:
                                shards[name] = stack.enter_context(safe_open(root / name, framework='pt', device='cpu'))
                            source = shards[name].get_tensor(key).view(torch.uint8)
                            if tuple(source.shape) != tuple(buffers[target][eid].shape):
                                raise ValueError(f'Native shape mismatch for {key}')
                            buffers[target][eid].copy_(source)
                del source
            return cls(*buffers).validate(experts, hidden, intermediate)
        finally:
            os.sched_setaffinity(0, previous)
