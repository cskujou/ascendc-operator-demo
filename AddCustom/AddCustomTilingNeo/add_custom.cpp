#include "kernel_operator.h"
#include "add_custom_tiling.h"

namespace AscendC {
constexpr int32_t BUFFER_NUM = 2;

class KernelAdd {
   private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<half> tensorX, tensorY, tensorZ;
    uint32_t tileNum;
    uint32_t blockLength;
    uint32_t tileLength;

    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);

   public:
    __aicore__ inline KernelAdd() = default;
    __aicore__ inline KernelAdd(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t tileNum);
    __aicore__ inline void Process();
};

__aicore__ inline KernelAdd::KernelAdd(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t tileNum) {
    this->tileNum = tileNum;
    this->blockLength = totalLength / GetBlockNum();
    this->tileLength = this->blockLength / tileNum / BUFFER_NUM;
    tensorX.SetGlobalBuffer((__gm__ half*)x + this->blockLength * GetBlockIdx(), this->blockLength);
    tensorY.SetGlobalBuffer((__gm__ half*)y + this->blockLength * GetBlockIdx(), this->blockLength);
    tensorZ.SetGlobalBuffer((__gm__ half*)z + this->blockLength * GetBlockIdx(), this->blockLength);
    pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(half));
    pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(half));
    pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(half));
}

__aicore__ inline void KernelAdd::Process() {
    for (int32_t i = 0; i < this->tileNum * BUFFER_NUM; i++) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
}

__aicore__ inline void KernelAdd::CopyIn(int32_t progress) {
    LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
    LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
    DataCopy(xLocal, tensorX[progress * this->tileLength], this->tileLength);
    DataCopy(yLocal, tensorY[progress * this->tileLength], this->tileLength);
    inQueueX.EnQue(xLocal);
    inQueueY.EnQue(yLocal);
}

__aicore__ inline void KernelAdd::Compute(int32_t progress) {
    LocalTensor<half> xLocal = inQueueX.DeQue<half>();
    LocalTensor<half> yLocal = inQueueY.DeQue<half>();
    LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
    Add(zLocal, xLocal, yLocal, this->tileLength);
    outQueueZ.EnQue(zLocal);
    inQueueX.FreeTensor(xLocal);
    inQueueY.FreeTensor(yLocal);
}

__aicore__ inline void KernelAdd::CopyOut(int32_t progress) {
    LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
    DataCopy(tensorZ[progress * this->tileLength], zLocal, this->tileLength);
    outQueueZ.FreeTensor(zLocal);
}
}  // namespace AscendC

extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, AddCustomTilingData tiling) {
    AscendC::KernelAdd op(x, y, z, tiling.totalLength, tiling.tileNum);
    op.Process();
}
