/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//!
//! SampleTest.cpp
//! This file contains the implementation of the MNIST API sample. It creates the network
//! for MNIST classification using the API.
//! It can be run with the following command line:
//! Command: ./sample_mnist_api [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]
//! [--useDLACore=<int>]
//!

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "logging.h"

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

const std::string gSampleName = "TensorRT.SampleTest";
Logger gLogger;

//!
//! \brief The SampleTestParams structure groups the additional parameters required by
//!         the SampleTest sample.
//!
struct SampleTestParams : public samplesCommon::SampleParams
{
    int x1,y1,z1,x2,y2,z2;
    // int inputH;                  //!< The input height
    // int inputW;                  //!< The input width
    // int outputSize;              //!< The output size
    // std::string weightsFile;     //!< The filename of the weights file
    // std::string mnistMeansProto; //!< The proto file containing means
};

//! \brief  The SampleTest class implements the MNIST API sample
//!
//! \details It creates the network for MNIST classification using the API
//!

nvinfer1::IPluginV2* PluginSlice( float x1)
{
    //Use the extern function getPluginRegistry to access the global TensorRT Plugin Registry
    auto creator = getPluginRegistry()->getPluginCreator("ResizeNearest_TRT2", "1");
    //populate the field parameters (say layerFields) for the plugin layer 
    nvinfer1::PluginField xx[1];

    xx[0]=nvinfer1::PluginField("scale",&x1,PluginFieldType::kFLOAT32);


    PluginFieldCollection pluginData;
    pluginData.fields=xx; 
    pluginData.nbFields=1;
    //create the plugin object using the layerName and the plugin meta data
    IPluginV2* pluginObj = creator->createPlugin("ResizeNearest_TRT2", &pluginData);
    assert(pluginObj);
    //add the plugin to the TensorRT network using the network API
    // auto layer = network.addPluginV2(&inputs[0], int(inputs.size()), pluginObj);
    // auto plugin_slice=new nvinfer1::plugin::Slice(x1,y1,zi,x2,y2,z2);
    return pluginObj;
}
class SampleTest
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    SampleTest(const SampleTestParams& params)
        : mParams(params)
        , mEngine(nullptr)
    {
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer();

    //!
    //! \brief Cleans up any state created in the sample class
    //!
    bool teardown();

private:
    SampleTestParams mParams; //!< The parameters for the sample.

    int mNumber{0}; //!< The number to classify

    std::map<std::string, nvinfer1::Weights> mWeightMap; //!< The weight name to weight value map

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    //!
    //! \brief Uses the API to create the MNIST Network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config);

    //!
    //! \brief Reads the input  and stores the result in a managed buffer
    //!
    bool processInput(const samplesCommon::BufferManager& buffers);

    //!
    //! \brief Classifies digits and verify result
    //!
    bool verifyOutput(const samplesCommon::BufferManager& buffers);

    //!
    //! \brief Loads weights from weights file
    //!
    std::map<std::string, nvinfer1::Weights> loadWeights(const std::string& file);
};

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the MNIST network by using the API to create a model and builds
//!          the engine that will be used to run MNIST (mEngine)
//!
//! \return Returns true if the engine was created successfully and false otherwise
//!
bool SampleTest::build()
{
    // mWeightMap = loadWeights(locateFile(mParams.weightsFile, mParams.dataDirs));

    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetwork());
    if (!network)
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto constructed = constructNetwork(builder, network, config);
    if (!constructed)
    {
        return false;
    }

    assert(network->getNbInputs() == 1);
    auto inputDims = network->getInput(0)->getDimensions();
    assert(inputDims.nbDims == 3);

    assert(network->getNbOutputs() == 1);
    auto outputDims = network->getOutput(0)->getDimensions();
    assert(outputDims.nbDims == 3);

    return true;
}

//!
//! \brief Uses the API to create the MNIST Network
//!
//! \param network Pointer to the network that will be populated with the MNIST network
//!
//! \param builder Pointer to the engine builder
//!
bool SampleTest::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config)
{
    // Create input tensor of shape { 1, 1, 28, 28 }
    ITensor* input1 = network->addInput(
        "input1", DataType::kFLOAT, Dims3{3,8,1});
    // ITensor* input2 = network->addInput(
    //     "input2", DataType::kFLOAT, Dims3{3, 8,1});
    // ITensor input_[2]={input1,input2};
    std::vector<ITensor*> inputs; 
    // inputs.push_back(input2);

    inputs.push_back(input1);

    auto slice=network->addPluginV2(&inputs[0],1,*PluginSlice(2.0));
    // slice->name="output"
    slice->getOutput(0)->setName(mParams.outputTensorNames[0].c_str());
    network->markOutput(*slice->getOutput(0));
    // network->markOutput(*prob->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(mParams.batchSize);
    config->setMaxWorkspaceSize(16_MiB);
    // mParams.fp16=true;
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllTensorScales(network.get(), 64.0f, 64.0f);
    }

    // samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool SampleTest::infer()
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine, mParams.batchSize);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    // Read the input data into the managed buffers
    assert(mParams.inputTensorNames.size() == 1);
    if (!processInput(buffers))
    {
        return false;
    }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    bool status = context->execute(mParams.batchSize, buffers.getDeviceBindings().data());

    if (!status)
    {
        return false;
    }

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();
    std::cout<<"copyOutputToHost ok "<<std::endl;
    // Verify results
    if (!verifyOutput(buffers))
    {
        return false;
    }

    return true;
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool SampleTest::processInput(const samplesCommon::BufferManager& buffers)
{
    // Read a random digit file
    srand(unsigned(time(nullptr)));
    std::vector<float> fileData(mParams.x1 * mParams.y1*mParams.z1);
    // mNumber = rand() % mParams.outputSize;
    // readPGMFile(locateFile(std::to_string(mNumber) + ".pgm", mParams.dataDirs), fileData.data(), mParams.inputH,
    //     mParams.inputW);

    // // Print ASCII representation of digit image
    // std::cout << "\nInput:\n" << std::endl;
    // for (int i = 0; i < mParams.inputH * mParams.inputW; i++)
    // {
    //     std::cout << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % mParams.inputW) ? "" : "\n");
    // }

    // Parse mean file
    // auto parser = SampleUniquePtr<nvcaffeparser1::ICaffeParser>(nvcaffeparser1::createCaffeParser());
    // if (!parser)
    // {
    //     return false;
    // }

    // auto meanBlob = SampleUniquePtr<nvcaffeparser1::IBinaryProtoBlob>(
    //     parser->parseBinaryProto(locateFile(mParams.mnistMeansProto, mParams.dataDirs).c_str()));
    // if (!meanBlob)
    // {
    //     return false;
    // }

    // const float* meanData = reinterpret_cast<const float*>(meanBlob->getData());
    // if (!meanData)
    // {
    //     return false;
    // }

    // Subtract mean from image
    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    for (int i = 0; i < mParams.x1 * mParams.y1*mParams.z1*mParams.batchSize; i++)
    {
        hostDataBuffer[i] = 55;
    }
    // hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[1]));
    // for (int i = 0; i < mParams.x2 * mParams.y2*mParams.z2; i++)
    // {
    //     hostDataBuffer[i] = 1;
    // }
    return true;
}

//!
//! \brief Classifies digits and verify result
//!
//! \return whether the classification output matches expectations
//!
bool SampleTest::verifyOutput(const samplesCommon::BufferManager& buffers)
{
    float* prob = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
    std::cout << "\nOutput:\n" << std::endl;
    float maxVal{0.0f};
    int idx{0};
    for (int i = 0; i < mParams.x1 * mParams.y1*mParams.z1*mParams.batchSize; i++)
    {
        
        std::cout << i << ": " << prob[i] << std::endl;
    }
    std::cout << std::endl;

    return idx == mNumber && maxVal > 0.9f;
}

//!
//! \brief Cleans up any state created in the sample class
//!
bool SampleTest::teardown()
{
    // Release weights host memory
    for (auto& mem : mWeightMap)
    {
        auto weight = mem.second;
        if (weight.type == DataType::kFLOAT)
        {
            delete[] static_cast<const uint32_t*>(weight.values);
        }
        else
        {
            delete[] static_cast<const uint16_t*>(weight.values);
        }
    }

    return true;
}

//!
//! \brief Loads weights from weights file
//!
//! \details TensorRT weight files have a simple space delimited format
//!          [type] [size] <data x size in hex>
//!
// std::map<std::string, nvinfer1::Weights> SampleTest::loadWeights(const std::string& file)
// {
//     gLogInfo << "Loading weights: " << file << std::endl;

//     // Open weights file
//     std::ifstream input(file, std::ios::binary);
//     assert(input.is_open() && "Unable to load weight file.");

//     // Read number of weight blobs
//     int32_t count;
//     input >> count;
//     assert(count > 0 && "Invalid weight map file.");

//     std::map<std::string, nvinfer1::Weights> weightMap;
//     while (count--)
//     {
//         nvinfer1::Weights wt{DataType::kFLOAT, nullptr, 0};
//         int type;
//         uint32_t size;

//         // Read name and type of blob
//         std::string name;
//         input >> name >> std::dec >> type >> size;
//         wt.type = static_cast<DataType>(type);

//         // Load blob
//         if (wt.type == DataType::kFLOAT)
//         {
//             uint32_t* val = new uint32_t[size];
//             for (uint32_t x = 0; x < size; ++x)
//             {
//                 input >> std::hex >> val[x];
//             }
//             wt.values = val;
//         }
//         else if (wt.type == DataType::kHALF)
//         {
//             uint16_t* val = new uint16_t[size];
//             for (uint32_t x = 0; x < size; ++x)
//             {
//                 input >> std::hex >> val[x];
//             }
//             wt.values = val;
//         }

//         wt.count = size;
//         weightMap[name] = wt;
//     }

//     return weightMap;
// }

//!
//! \brief Initializes members of the params struct using the command line args
//!
SampleTestParams initializeSampleParams(const samplesCommon::Args& args)
{
    SampleTestParams params;
    if (args.dataDirs.empty()) //!< Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("data/mnist/");
        params.dataDirs.push_back("data/samples/mnist/");
    }
    else //!< Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }
    params.inputTensorNames.push_back("input1");
    // params.inputTensorNames.push_back("input2");

    params.batchSize = 2;
    params.outputTensorNames.push_back("prob");
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;

    params.x1 = 3;
    params.y1 = 8;
    params.z1=1;
    params.x2 = 3;
    params.y2 = 8;
    params.z2=1;

    return params;
}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout
        << "Usage: ./sample_mnist_api [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]"
        << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "(data/samples/mnist/, data/mnist/)"
              << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform."
              << std::endl;
    std::cout << "--int8          Run in Int8 mode." << std::endl;
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
}

int main(int argc, char** argv)
{
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        // gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    auto sampleTest = gLogger.defineTest(gSampleName, argc, argv);

    gLogger.reportTestStart(sampleTest);

    SampleTest sample(initializeSampleParams(args));

    // gLogInfo << "Building and running a GPU inference engine for MNIST API" << std::endl;

    if (!sample.build())
    {
        return gLogger.reportFail(sampleTest);
    }
    if (!sample.infer())
    {
        return gLogger.reportFail(sampleTest);
    }
    // if (!sample.teardown())
    // {
    //     return gLogger.reportFail(sampleTest);
    // }

    return gLogger.reportPass(sampleTest);
}
