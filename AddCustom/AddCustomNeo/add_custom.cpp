#include "kernel_operator.h"

namespace AscendC {
constexpr int32_t TOTAL_LENGTH = 8 * 2048;
constexpr int32_t USE_CORE_NUM = 8;
constexpr int32_t BLOCK_LENGTH = TOTAL_LENGTH / USE_CORE_NUM;
constexpr int32_t TILE_NUM = 8;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t TILE_LENGTH = BLOCK_LENGTH / TILE_NUM / BUFFER_NUM;

class KernelAdd {
   private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<half> tensorX, tensorY, tensorZ;

    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);

   public:
    __aicore__ inline KernelAdd() = default;
    __aicore__ inline KernelAdd(GM_ADDR x, GM_ADDR y, GM_ADDR z);
    __aicore__ inline void Process();
};

__aicore__ inline KernelAdd::KernelAdd(GM_ADDR x, GM_ADDR y, GM_ADDR z) {
    tensorX.SetGlobalBuffer((__gm__ half*)x + BLOCK_LENGTH * GetBlockIdx(), BLOCK_LENGTH);
    tensorY.SetGlobalBuffer((__gm__ half*)y + BLOCK_LENGTH * GetBlockIdx(), BLOCK_LENGTH);
    tensorZ.SetGlobalBuffer((__gm__ half*)z + BLOCK_LENGTH * GetBlockIdx(), BLOCK_LENGTH);
    pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(half));
    pipe.InitBuffer(inQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(half));
    pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LENGTH * sizeof(half));
}

__aicore__ inline void KernelAdd::Process() {
    for (int32_t i = 0; i < TILE_NUM * BUFFER_NUM; i++) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
}

__aicore__ inline void KernelAdd::CopyIn(int32_t progress) {
    LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
    LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
    DataCopy(xLocal, tensorX[progress * TILE_LENGTH], TILE_LENGTH);
    DataCopy(yLocal, tensorY[progress * TILE_LENGTH], TILE_LENGTH);
    inQueueX.EnQue(xLocal);
    inQueueY.EnQue(yLocal);
}

__aicore__ inline void KernelAdd::Compute(int32_t progress) {
    LocalTensor<half> xLocal = inQueueX.DeQue<half>();
    LocalTensor<half> yLocal = inQueueY.DeQue<half>();
    LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
    Add(zLocal, xLocal, yLocal, TILE_LENGTH);
    outQueueZ.EnQue(zLocal);
    inQueueX.FreeTensor(xLocal);
    inQueueY.FreeTensor(yLocal);
}

__aicore__ inline void KernelAdd::CopyOut(int32_t progress) {
    LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
    DataCopy(tensorZ[progress * TILE_LENGTH], zLocal, TILE_LENGTH);
    outQueueZ.FreeTensor(zLocal);
}
}  // namespace AscendC

extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z) {
    AscendC::KernelAdd op(x, y, z);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void add_custom_do(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* z) {
    add_custom<<<blockDim, nullptr, stream>>>(x, y, z);
}
#endif
