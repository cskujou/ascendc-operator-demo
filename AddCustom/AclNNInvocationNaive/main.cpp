/**
 * @file main.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "acl/acl.h"
#include "aclnn_add_custom.h"

#define SUCCESS 0
#define FAILED 1

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
    // Fixed code, acl initialization
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return FAILED);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return FAILED);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return FAILED);

    return SUCCESS;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate device memory
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return FAILED);

    // Call aclrtMemcpy to copy host data to device memory
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return FAILED);

    // Call aclCreateTensor to create a aclTensor object
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    return SUCCESS;
}

void DestroyResources(std::vector<void*> tensors, std::vector<void*> deviceAddrs, aclrtStream stream, int32_t deviceId,
                      void* workspaceAddr = nullptr) {
    // Release aclTensor and device
    for (uint32_t i = 0; i < tensors.size(); i++) {
        if (tensors[i] != nullptr) {
            aclDestroyTensor(reinterpret_cast<aclTensor*>(tensors[i]));
        }
        if (deviceAddrs[i] != nullptr) {
            aclrtFree(deviceAddrs[i]);
        }
    }
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    // Destroy stream and reset device
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

template <typename T>
void GenerateRandomHostData(std::vector<T>& hostData, float minVal = 0.0f, float maxVal = 10.0f) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<float> distribution(minVal, maxVal);  // 均匀分布在 [minVal, maxVal) 之间

    for (auto& data : hostData) {
        float randomFloat = distribution(generator);
        if constexpr (std::is_same<T, aclFloat16>::value) {
            data = aclFloatToFloat16(randomFloat);
        } else {
            // 其他数值类型直接转换赋值（如 float/double/int 等）
            data = static_cast<T>(randomFloat);
        }
    }
}

int main(int argc, char** argv) {
    // 1. (Fixed code) Initialize device / stream, refer to the list of external interfaces of acl
    // Update deviceId to your own device id
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return FAILED);

    // 2. Create input and output, need to customize according to the interface of the API
    std::vector<int64_t> inputXShape = {8, 2048};
    std::vector<int64_t> inputYShape = {8, 2048};
    std::vector<int64_t> outputZShape = {8, 2048};
    void* inputXDeviceAddr = nullptr;
    void* inputYDeviceAddr = nullptr;
    void* outputZDeviceAddr = nullptr;
    aclTensor* inputX = nullptr;
    aclTensor* inputY = nullptr;
    aclTensor* outputZ = nullptr;

    std::vector<aclFloat16> inputXHostData(inputXShape[0] * inputXShape[1]);
    std::vector<aclFloat16> inputYHostData(inputYShape[0] * inputYShape[1]);
    std::vector<aclFloat16> outputZHostData(outputZShape[0] * outputZShape[1]);

    GenerateRandomHostData(inputXHostData, -100.0f, 100.0f);
    GenerateRandomHostData(inputYHostData, -100.0f, 100.0f);
    std::fill(outputZHostData.begin(), outputZHostData.end(), aclFloatToFloat16(0.0f));
    // for (int i = 0; i < inputXShape[0] * inputXShape[1]; ++i) {
    //     inputXHostData[i] = aclFloatToFloat16(1.0);
    //     inputYHostData[i] = aclFloatToFloat16(2.0);
    //     outputZHostData[i] = aclFloatToFloat16(0.0);
    // }
    std::vector<void*> tensors = {inputX, inputY, outputZ};
    std::vector<void*> deviceAddrs = {inputXDeviceAddr, inputYDeviceAddr, outputZDeviceAddr};
    // Create inputX aclTensor
    ret = CreateAclTensor(inputXHostData, inputXShape, &inputXDeviceAddr, aclDataType::ACL_FLOAT16, &inputX);
    CHECK_RET(ret == ACL_SUCCESS, DestroyResources(tensors, deviceAddrs, stream, deviceId); return FAILED);
    // Create inputY aclTensor
    ret = CreateAclTensor(inputYHostData, inputYShape, &inputYDeviceAddr, aclDataType::ACL_FLOAT16, &inputY);
    CHECK_RET(ret == ACL_SUCCESS, DestroyResources(tensors, deviceAddrs, stream, deviceId); return FAILED);
    // Create outputZ aclTensor
    ret = CreateAclTensor(outputZHostData, outputZShape, &outputZDeviceAddr, aclDataType::ACL_FLOAT16, &outputZ);
    CHECK_RET(ret == ACL_SUCCESS, DestroyResources(tensors, deviceAddrs, stream, deviceId); return FAILED);

    // 3. Call the API of the custom operator library
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Calculate the workspace size and allocate memory for it
    ret = aclnnAddCustomGetWorkspaceSize(inputX, inputY, outputZ, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddCustomGetWorkspaceSize failed. ERROR: %d\n", ret);
              DestroyResources(tensors, deviceAddrs, stream, deviceId); return FAILED);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                  DestroyResources(tensors, deviceAddrs, stream, deviceId, workspaceAddr); return FAILED);
    }
    // Execute the custom operator
    ret = aclnnAddCustom(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdd failed. ERROR: %d\n", ret);
              DestroyResources(tensors, deviceAddrs, stream, deviceId, workspaceAddr); return FAILED);

    // 4. (Fixed code) Synchronize and wait for the task to complete
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
              DestroyResources(tensors, deviceAddrs, stream, deviceId, workspaceAddr); return FAILED);

    // 5. Get the output value, copy the result from device memory to host memory, need to modify according to the
    // interface of the API
    auto size = GetShapeSize(outputZShape);
    std::vector<aclFloat16> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outputZDeviceAddr,
                      size * sizeof(aclFloat16), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
              DestroyResources(tensors, deviceAddrs, stream, deviceId, workspaceAddr); return FAILED);

    // 6. Detroy resources, need to modify according to the interface of the API
    DestroyResources(tensors, deviceAddrs, stream, deviceId, workspaceAddr);

    // print the output result
    // std::vector<aclFloat16> goldenData(size, aclFloatToFloat16(3.0));
    std::vector<aclFloat16> goldenData(size);
    for (int64_t i = 0; i < size; ++i) {
        // golden = inputX + inputY（float 精度计算，避免 float16 精度损失）
        float x = aclFloat16ToFloat(inputXHostData[i]);
        float y = aclFloat16ToFloat(inputYHostData[i]);
        goldenData[i] = aclFloatToFloat16(x + y);
    }

    LOG_PRINT("result is:\n");
    for (int64_t i = 0; i < 10; i++) {
        LOG_PRINT("%.1f ", aclFloat16ToFloat(resultData[i]));
    }

    bool testPass = true;
    const float eps = 1e-3f;  // 精度容忍度（可根据需求调整）
    for (int64_t i = 0; i < size; ++i) {
        float result = aclFloat16ToFloat(resultData[i]);
        float golden = aclFloat16ToFloat(goldenData[i]);
        if (std::fabs(result - golden) > eps) {
            testPass = false;
            LOG_PRINT("mismatch at idx%d: result=%.4f, golden=%.4f\n", i, result, golden);
            break;  // 发现不匹配即退出验证
        }
    }
    LOG_PRINT("\n");
    if (testPass) {
        LOG_PRINT("test pass\n");
    } else {
        LOG_PRINT("test failed\n");
        return FAILED;
    }
    return SUCCESS;
}
