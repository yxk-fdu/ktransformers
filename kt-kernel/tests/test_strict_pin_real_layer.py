import gc
from kt_kernel.utils.llamafile import LlamafileMoEWrapper

def locked():
    with open('/proc/self/status') as f:
        return int(next(x for x in f if x.startswith('VmLck:')).split()[1])

baseline = locked()
w = LlamafileMoEWrapper(layer_idx=0, num_experts=256, num_experts_per_tok=1,
    hidden_size=4096, moe_intermediate_size=2048, gpu_experts_mask=None, cpuinfer_threads=4,
    threadpool_count=1, numa_nodes=[1], weight_path='/data/weight/cpu_moe/dsv4_layer0_mxfp4.gguf',
    chunked_prefill_size=1)
w.load_weights(None)
expected = sum(t.numel() * t.element_size() for t in w.weights_to_keep)
actual = (locked() - baseline) * 1024
assert actual >= expected, (actual, expected)
assert all(t.device.type == 'cpu' for t in w.weights_to_keep)
print('PASS real MXFP4 layer: weight_bytes=', expected, 'locked_bytes=', actual, flush=True)
import re, collections
large = set()
for block in re.split(r'(?=^[0-9a-f]+-[0-9a-f]+ )', open('/proc/self/smaps').read(), flags=re.M):
    if re.search(r'^VmFlags:.*\blo\b', block, re.M):
        size = re.search(r'^Anonymous:\s+(\d+)', block, re.M)
        if size and int(size.group(1)) > 1024 * 1024:
            large.add(block.split('-', 1)[0])
nodes = collections.Counter()
for line in open('/proc/self/numa_maps'):
    if line.split()[0] in large:
        for n, pages in re.findall(r'\bN(\d+)=(\d+)', line):
            nodes[int(n)] += int(pages)
assert nodes and nodes[1] / sum(nodes.values()) > 0.999, nodes
print('PASS real layer NUMA1 placement', dict(nodes), flush=True)

del w
gc.collect()
assert locked() == baseline
print('PASS real layer lock cleanup', flush=True)
