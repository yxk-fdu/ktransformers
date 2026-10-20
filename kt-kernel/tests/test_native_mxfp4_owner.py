import json
import os
import pytest
import torch
from safetensors.torch import save_file
from kt_kernel.utils.native_mxfp4 import NativeMXFP4Weights


def make_owner():
    return NativeMXFP4Weights(
        torch.zeros(3, 256, 128, dtype=torch.uint8),
        torch.zeros(3, 256, 128, dtype=torch.uint8),
        torch.zeros(3, 256, 128, dtype=torch.uint8),
        torch.full((3, 256, 8), 127, dtype=torch.uint8),
        torch.full((3, 256, 8), 127, dtype=torch.uint8),
        torch.full((3, 256, 8), 127, dtype=torch.uint8),
    )


def test_owner_validation_returns_same_storage():
    owner = make_owner()
    pointers = [x.data_ptr() for x in owner.tensors]
    assert owner.validate(3, 256, 256) is owner
    assert [x.data_ptr() for x in owner.tensors] == pointers
    assert owner.nbytes == 3 * 3 * 256 * 256 * 17 // 32


def test_owner_rejects_noncontiguous_or_wrong_dtype():
    owner = make_owner()
    owner.gate = owner.gate.transpose(1, 2)
    with pytest.raises(ValueError, match='contiguous'):
        owner.validate(3, 256, 256)
    owner = make_owner()
    owner.gate = owner.gate.float()
    with pytest.raises(ValueError, match='uint8'):
        owner.validate(3, 256, 256)


def test_checkpoint_load_preserves_native_bytes_and_affinity(tmp_path):
    tensors = {}
    mapping = {}
    for e in range(3):
        for proj in ['w1', 'w3', 'w2']:
            for suffix, shape in [('weight', (256, 128)), ('scale', (256, 8))]:
                key = f'layers.0.ffn.experts.{e}.{proj}.{suffix}'
                tensors[key] = torch.full(shape, 127 if suffix == 'scale' else e + 1, dtype=torch.uint8)
                mapping[key] = 'test.safetensors'
    save_file(tensors, tmp_path / 'test.safetensors')
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': mapping}))
    before = os.sched_getaffinity(0)
    owner = NativeMXFP4Weights.from_checkpoint(tmp_path, 0, 3, 256, 256, numa_node=1)
    assert os.sched_getaffinity(0) == before
    for e in range(3):
        for tensor in owner.tensors[:3]:
            assert torch.equal(tensor[e], torch.full((256, 128), e + 1, dtype=torch.uint8))


def test_reserved_scale_raises_before_worker_dispatch():
    from kt_kernel import kt_kernel_ext as ext
    owner = make_owner()
    owner.gate_scale[0, 0, 0] = 255
    wp = ext.WorkerPoolConfig()
    wp.subpool_count = 1
    wp.subpool_numa_map = [1]
    wp.subpool_thread_count = [1]
    pool = ext.CPUInfer(wp)
    cfg = ext.moe.MOEConfig(3, 3, 256, 256)
    cfg.pool = pool.backend_
    cfg.m_block = 32
    cfg.max_len = 1
    cfg.group_min_len = 10
    cfg.group_max_len = 1
    cfg.gate_type = cfg.up_type = cfg.down_type = 39
    cfg.hidden_type = 30
    cfg.native_mxfp4 = True
    for field, tensor in zip(['gate_proj', 'up_proj', 'down_proj', 'gate_scale', 'up_scale', 'down_scale'],
                             owner.tensors):
        setattr(cfg, field, tensor.data_ptr())
    with pytest.raises(ValueError, match='Reserved E8M0'):
        ext.moe.MOE(cfg)
