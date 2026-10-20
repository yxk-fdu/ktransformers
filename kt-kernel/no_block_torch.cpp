#if defined(USE_ASCEND_NPU)
#include <cstdint>
#include <acl/acl.h>
#include "torch_npu/csrc/framework/OpCommand.h"
extern "C" void launch_no_block_torch(uintptr_t callback, uintptr_t data, uintptr_t stream) {
    at_npu::native::OpCommand::RunOpApiV2("CpuMoENoBlockCallback", [=]() -> int {
        return aclrtLaunchCallback(reinterpret_cast<aclrtCallback>(callback),
            reinterpret_cast<void*>(data), ACL_CALLBACK_NO_BLOCK,
            reinterpret_cast<aclrtStream>(stream));
    });
}

#endif
