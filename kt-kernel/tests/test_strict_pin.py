"""Run with PYTHONPATH pointing at the newly built extension directory."""
import ctypes
import tempfile
import gc
import mmap
import resource
import sys
import numpy as np
import kt_kernel_ext as ext


def locked_kb():
    with open('/proc/self/status') as f:
        return int(next(line for line in f if line.startswith('VmLck:')).split()[1])


def exercise(parts):
    baseline = locked_kb()
    wc = ext.WorkerPoolConfig()
    wc.subpool_count = parts
    wc.subpool_numa_map = [1] if parts == 1 else [1, 3]
    wc.subpool_thread_count = [2] * parts
    pool = ext.CPUInfer(wc)
    import os, re
    for tid in os.listdir('/proc/self/task'):
        status = open('/proc/self/task/' + tid + '/status').read()
        name = re.search(r'^Name:\s+(.*)', status, re.M).group(1)
        match = re.match(r'numa_(\d+)_[tm]_', name)
        if match:
            node = int(match.group(1))
            expected_cpus = set()
            for token in open(f'/sys/devices/system/node/node{node}/cpulist').read().strip().split(','):
                bounds = [int(v) for v in token.split('-')]
                expected_cpus.update(range(bounds[0], bounds[-1] + 1))
            assert os.sched_getaffinity(int(tid)) <= expected_cpus, (name, os.sched_getaffinity(int(tid)))

    h, intermediate, experts = 256, 512, 2
    nbytes = h * intermediate * experts * 4
    files = [tempfile.TemporaryFile() for _ in range(3)]
    for f in files:
        f.truncate(nbytes)
    maps = [mmap.mmap(f.fileno(), nbytes) for f in files]
    arrays = [np.ndarray((experts, intermediate, h), dtype=np.float32, buffer=m) for m in maps]
    rng = np.random.default_rng(17)
    for a in arrays:
        a[:] = rng.normal(0, 0.01, a.shape)
    cfg = ext.moe.MOEConfig(experts, 1, h, intermediate)
    cfg.pool = pool.backend_
    cfg.gate_type = cfg.up_type = cfg.down_type = 0  # GGML F32
    cfg.hidden_type = 30  # GGML BF16
    cfg.m_block = 32
    cfg.group_min_len = 10
    cfg.group_max_len = cfg.max_len = 1
    cfg.gate_proj, cfg.up_proj, cfg.down_proj = [a.ctypes.data for a in arrays]
    moe = ext.moe.MOE(cfg)
    try:
        moe.pin_weights()
        raise AssertionError('pin-before-load must fail')
    except RuntimeError as e:
        assert 'before pinning' in str(e)
    pool.submit(moe.load_weights_task())
    pool.sync()
    qlen = np.array([1], dtype=np.int32)
    ids = np.array([0], dtype=np.int64)
    routing = np.array([1.0], dtype=np.float32)
    x = np.full(h, 0x3f80, dtype=np.uint16)  # BF16 1.0
    def forward():
        out = np.zeros(h, dtype=np.uint16)
        pool.submit(moe.forward_task(qlen.ctypes.data, 1, ids.ctypes.data, routing.ctypes.data, x.ctypes.data, out.ctypes.data))
        pool.sync()
        return out
    before = forward()
    if '--failure' in sys.argv:
        class Header(ctypes.Structure):
            _fields_ = [('version', ctypes.c_uint32), ('pid', ctypes.c_int)]
        class Data(ctypes.Structure):
            _fields_ = [('effective', ctypes.c_uint32), ('permitted', ctypes.c_uint32), ('inheritable', ctypes.c_uint32)]
        libc = ctypes.CDLL(None, use_errno=True)
        header, data = Header(0x20080522, 0), (Data * 2)()
        assert libc.capget(ctypes.byref(header), data) == 0
        data[0].effective &= ~(1 << 14)  # CAP_IPC_LOCK
        data[0].permitted &= ~(1 << 14)
        assert libc.capset(ctypes.byref(header), data) == 0
        limit = nbytes + 4096 if '--partial' in sys.argv else 0
        resource.setrlimit(resource.RLIMIT_MEMLOCK, (limit, limit))
        try:
            moe.pin_weights()
            raise AssertionError('mlock must fail without privilege or quota')
        except RuntimeError as e:
            assert 'mlock failed' in str(e)
        assert locked_kb() == baseline
    else:
        moe.pin_weights()
        assert locked_kb() >= baseline + 3 * nbytes // 1024
        count = locked_kb()
        moe.pin_weights()
        assert locked_kb() == count
        np.testing.assert_array_equal(before, forward())
    del moe
    gc.collect()
    assert locked_kb() == baseline, (locked_kb(), baseline)
    print('PASS parts=', parts, 'failure=', '--failure' in sys.argv)

exercise(int(sys.argv[1]))
