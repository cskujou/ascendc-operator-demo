#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <string>

#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
using namespace matmul_tiling;
using namespace std;

/**
 * @brief  Generate matmul tiling.
 * @param  socVersion: Platform socversion.
 * @param  tilingBuf data buffer.
 */
void GenerateTiling(const char* socVersion, uint8_t* tilingBuf) {
    constexpr int32_t M = 2048;
    constexpr int32_t N = 1024;
    constexpr int32_t K = 512;

    TPosition leftPosition = TPosition::GM;
    CubeFormat leftFormat = CubeFormat::ND;
    DataType leftDtype = DataType::DT_FLOAT16;
    bool isTransA = false;

    TPosition rightPosition = TPosition::GM;
    CubeFormat rightFormat = CubeFormat::ND;
    DataType rightDtype = DataType::DT_FLOAT16;
    bool isTransB = false;

    TPosition resultPosition = TPosition::GM;
    CubeFormat resultFormat = CubeFormat::ND;
    DataType resultDtype = DataType::DT_FLOAT;

    bool isBias = false;

    constexpr int32_t SINGLECORE_M = 512;
    constexpr int32_t SINGLECORE_N = 512;

    optiling::TCubeTiling tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
    MultiCoreMatmulTiling tilingApi(*ascendcPlatform);

    tilingApi.SetDim(
        ascendcPlatform->GetCoreNumAiv());
    tilingApi.SetAType(leftPosition, leftFormat, leftDtype, isTransA);
    tilingApi.SetBType(rightPosition, rightFormat, rightDtype, isTransB);
    tilingApi.SetCType(resultPosition, resultFormat, resultDtype);

    tilingApi.SetOrgShape(M, N, K);
    tilingApi.SetShape(M, N, K);
    if (ascendcPlatform->GetSocVersion() == platform_ascendc::SocVersion::ASCEND310P) {
        tilingApi.SetSingleShape(SINGLECORE_M, SINGLECORE_N, -1);
        int32_t mBlockNum = M / SINGLECORE_M;
        int32_t nBlockNum = N / SINGLECORE_N;
        tilingApi.SetDim(mBlockNum * nBlockNum);
    }
    tilingApi.SetBias(isBias);
    tilingApi.SetBufferSpace(-1, -1, -1);

    int64_t res = tilingApi.GetTiling(tilingData);
    if (res == -1) {
        std::cout << "gen tiling failed" << std::endl;
    }
    uint32_t tcubeTilingSize = tilingData.GetDataSize();
    tilingData.SaveToBuffer(tilingBuf, tcubeTilingSize);

    uint64_t localMemSize;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, localMemSize);
    *reinterpret_cast<uint64_t*>(tilingBuf + tcubeTilingSize) = localMemSize;
    return;
}
