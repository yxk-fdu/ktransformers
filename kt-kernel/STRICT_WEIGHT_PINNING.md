# Strict CPU expert DRAM residency

`KT_MOE_PIN_WEIGHTS=1` requires successful `mlock` of every final gate/up/down weight range after CPUInfer loading is synchronized. Strict mode copies single- and multi-NUMA weights into C++ buffers first-touched by the configured CPU subpool, then locks those final buffers. Without strict pinning, the single-NUMA mmap alias optimization remains available. Source tensors remain owned by the wrapper, and the GGUFLoader owns the mappings. No Python integer pointers are passed to libc.

A successful load means every weight page is resident and cannot be reclaimed or swapped until unlocked, unmapped or process exit. It does not promise zero minor faults for unrelated buffers. Lock failure raises a synchronous RuntimeError with layer, NUMA part, projection, byte count and errno; all partially acquired locks are rolled back and the wrapper disables its kernel. Configure sufficient RLIMIT_MEMLOCK or CAP_IPC_LOCK in the actual worker process. No host swap settings need changing.

The kernel destructor unlocks weights and deletes only its own multi-NUMA buffers; aliased mapped weights are never deleted. Repeated pin calls are idempotent. Wrapper reload is rejected to avoid replacing live pointer owners. Each model layer should have one wrapper owning its weight locking lifecycle.

Validation on server 141.61.141.22 in cpu_moe:
- SVE=OFF extension rebuilt successfully with existing staged hwloc and Python headers.
- tests/test_strict_pin.py: file-mapped single-NUMA and copied multi-NUMA paths; bitwise identical CPU forward before/after locking; lock count; repeated pin; destruction; zero quota failure; partial lock rollback. All passed.
- tests/test_strict_pin_real_layer.py with KT_MOE_PIN_WEIGHTS=1: real dsv4_layer0_mxfp4.gguf, 256 experts, 4096 hidden, 2048 intermediate. 3422552064 weight bytes; 3422556160 locked bytes after page alignment. Cleanup restored VmLck baseline.
- All three CPU MoE launch scripts pass bash -n and enable strict locking. Eager/graph default KT_BUILD now points at the tested workspace package because /root/ktbuild/lib is absent on this server; explicit overrides remain supported.

Old binary and installed loader backups have suffix `.pre-strict-pin-20261006`. Binary installation used atomic replacement, preserving existing mapped binaries in running processes. Existing services were not restarted. Full-model serving has not been run as part of this change.

## NUMA alignment update (2026-10-06)

The graph script now selects physical NUMA node 1. Fixed out-of-bounds per-node bookkeeping for subpool_count=1 with subpool_numa_map=[1], using maps keyed by physical node instead of vectors sized by pool count. The distributor memory policy now resolves the logical subpool index through numa_ids.

Regression covered physical nodes [1] and [1,3], CPU output parity, pin failure and partial rollback. Real layer0 MXFP4 validation passed: all 835587 locked anonymous pages were on NUMA 1; cleanup restored VmLck.
