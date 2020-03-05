#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include<memory>
#include <sys/stat.h>
#include <unordered_set>
#include <vector>
#include <array>
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include <NvInferRuntimeCommon.h>
#include "NvCaffeParser.h"
#include "NvInfer.h"
#include <cuda_runtime_api.h>
const std::string gSampleName = "test_slice";
std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
nvinfer1::IPluginV2IOExt* PluginSlice(const int x1,const int y1,const int z1,const int x2,const int y2,const int z2)
{
    auto plugin_slice=new nvinfer1::plugin::Slice(x1,y1,zi,x2,y2,z2);
    return plugin_slice;
}
bool constructNetwork(nvinfer1::IBuilder* builder,
    nvinfer1::INetworkDefinition* network, nvinfer1::IBuilderConfig* config)
{
    
    builder->setMaxBatchSize(1);
    config->setMaxWorkspaceSize(1024*1024);
    config->setFlag(BuilderFlag::kFP16);
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    return true;
}

bool build()
{
    IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetwork();
    auto config = builder->createBuilderConfig();
    auto constructed = constructNetwork(builder, network, config);
    auto inputDims = network->getInput(0)->getDimensions();
    auto outputDims = network->getOutput(0)->getDimensions();
    return true;
}

bool processInput(const samplesCommon::BufferManager& buffers)
{
    std::vector<float> data1,data2;
    float* hostDataBuffer1 = static_cast<float*>(buffers.getHostBuffer("input1"));
    float* hostDataBuffer2 = static_cast<float*>(buffers.getHostBuffer("input2"));

    for(int i=0;i<8*3;i++)
    {
        data1.push_back(0);
        hostDataBuffer1[i]=0;
    }
    for(int i=0;i<3*8;i++)
    {
        data2.push_back(0);
        hostDataBuffer2[i]=0;

    }
    return 0;
}
bool verifyOutput(const samplesCommon::BufferManager& buffers)
{
    float* prob = static_cast<float*>(buffers.getHostBuffer("output"));
    std::cout << "\nOutput:\n" << std::endl;
    float maxVal{0.0f};
    int idx{0};
    for (int i = 0; i < 3*8; i++)
    {
        
        std::cout << i << ": " << prob[i]<< std::endl;
    }
    std::cout << std::endl;
}
bool infer()
{
    samplesCommon::BufferManager buffers(mEngine, 1);
    auto context = mEngine->createExecutionContext();
    processInput(buffers);
    buffers.copyInputToDevice();
    bool status = context->execute(1, buffers.getDeviceBindings().data());
    buffers.copyOutputToHost();
    verifyOutput(buffers);
    return true;
}


int main(int argc, char** argv)
{
    auto sampleTest = gLogger.defineTest(gSampleName, argc, argv);

    return 0;
}