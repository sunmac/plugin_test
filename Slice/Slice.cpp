#include "Slice.h"
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>
#include<NvInferRuntimeCommon.h>


#include "plugin.h"

// using namespace nvinfer1;

//using nvinfer1::PluginField;
//using nvinfer1::PluginFieldCollection;

using nvinfer1::plugin::Slice;
using nvinfer1::plugin::SliceCreator;
nvinfer1::PluginFieldCollection SliceCreator::mFC{};
std::vector<nvinfer1::PluginField> nvinfer1::plugin::SliceCreator::mPluginAttributes;
namespace
{
const char* SLICE_PLUGIN_VERSION{"1"};
const char* SLICE_PLUGIN_NAME{"LICE_TRT"};
}



Slice::Slice(int x1,int y1,int z1,int x2,int y2,int z2,nvinfer1::DataType inputtpye):x1(x1),y1(y1),z1(z1),x2(x2),y2(y2),z2(z2),mType(inputtpye){}
Slice::Slice(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    
    x1 = read<int>(d);
    y1 = read<int>(d);
    z1 = read<int>(d);
    x2 = read<int>(d);
    y2 = read<int>(d);
    z2 = read<int>(d);
    mType=read<nvinfer1::DataType>(d);
    ASSERT(d == a + length);
}
int Slice::getNbOutputs() const
{
    // 有几个输出
    return 1;
}
int Slice::initialize()
{
    //如果plugin计算中间过程需要变量或者空间时，通过initialize进行初始化
    return STATUS_SUCCESS;
}
void Slice::terminate() {}//释放空间



nvinfer1::Dims Slice::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    //assert input数据尺寸 inputs
    //assert input变量个数 nbInputDims
    //index?
    ASSERT(nbInputDims == 2);
    // std::cout<<"inputs[1].d[0]"<<inputs[1].d[0]<<inputs[1].d[1]<<inputs[1].d[2]<<std::endl;
    return inputs[1];
}


size_t Slice::getWorkspaceSize(int maxBatchSize) const
{
    //gpu显存需要工作空间的大小
    size_t sum_memory=x1*y1*z1*2+x2*y2*z2;
    return 0;
}
int Slice::enqueue(
    int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
    {
    //plugin中核心算法的代码，指定输入、输出和流。
        switch (mType)
        {
            case DataType::kFLOAT:
            {
                const float* input_data = static_cast<const float*>(inputs[0]);
                const float* input_slice = static_cast<const float*>(inputs[1]);
                float* inputTemp= static_cast<float*>(outputs[0]);
                std::cout<<"batchSize "<<batchSize<<std::endl;
                pluginStatus_t status = Sliceinference( stream, x1, y1, z1, x2, y2, z2, input_data,input_slice,inputTemp,batchSize);
                ASSERT(status == STATUS_SUCCESS);
                break;
            }
            case DataType::kHALF:
            {
                const half* input_data = static_cast<const half*>(inputs[0]);
                const half* input_slice = static_cast<const half*>(inputs[1]);
                half* inputTemp= static_cast<half*>(outputs[0]);
                //cudaMalloc(&inputTemp, sizeof(input_data));
            // std::cout<<batchSize<<std::endl;
                pluginStatus_t status = Sliceinference( stream, x1, y1, z1, x2, y2, z2, input_data,input_slice,inputTemp,batchSize);
                ASSERT(status == STATUS_SUCCESS);

                break;
            }
        }
    
    
    return cudaGetLastError() != cudaSuccess;;
}
size_t Slice::getSerializationSize() const
{
    //参数的个数 getSerializationSize给serialize提供buffer的尺寸
    size_t type_size=(mType == nvinfer1::DataType::kFLOAT ? sizeof(float) : sizeof(__half));
    return 6*sizeof(int)+type_size;
}
void Slice::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write<int>(d,x1);
    write<int>(d,y1);
    write<int>(d,z1);
    write<int>(d,x2);
    write<int>(d,y2);
    write<int>(d,z2);
    write(d, mType);

    ASSERT(d == a + getSerializationSize());

    //ASSERT(d == a + getSerializationSize());
}
void Slice::configurePlugin(const PluginTensorDesc * in,int nbInput,const PluginTensorDesc * out,int nbOutput )
{
    // 传送输入和输出的数量、所有输入和输出的维度和数据类型、所有输入和输出的广播信息、所选择的插件格式和最大批量大小。此时，插件将设置其内部状态，并为给定的配置。
    ASSERT(nbInput == 2);
    ASSERT(nbOutput == 1);
    std::cout<<in[0].dims.nbDims<<std::endl;
    // ASSERT(in[0].dims.nbDims == 3);
    // ASSERT(in[0].type==nvinfer1::DataType::kFLOAT ||in[0].type==nvinfer1::DataType::kHALF)
    // ASSERT(in[1].dims.nbDims == 3);
    // ASSERT(in[1].type==nvinfer1::DataType::kFLOAT ||in[1].type==nvinfer1::DataType::kHALF)
}

// bool Slice::supportsFormat(DataType type, PluginFormat format) const     //ipluginv2ext
// {
//     //检查 plugin支持的数据类型  PluginFormat::kNCHW
//     return ((type == DataType::kFLOAT || type == DataType::kINT32) );
// }
bool Slice::supportsFormatCombination(int pos,const PluginTensorDesc* inOut,int nbInputs,int nbOutputs)const
{
    return inOut[pos].format==PluginFormat::kNCHW&&(inOut[pos].type==DataType::kFLOAT||inOut[pos].type==DataType::kHALF); 
}
const char* Slice::getPluginType() const
{
    // plugin name
    return SLICE_PLUGIN_NAME;
}

const char* Slice::getPluginVersion() const
{
    // plugin version
    return SLICE_PLUGIN_VERSION;
}

void Slice::destroy()
{
    delete this;
}

nvinfer1::IPluginV2IOExt* Slice::clone() const
{
    IPluginV2IOExt* plugin = new Slice(x1,y1,z1,x2,y2,z2,mType);
    plugin->setPluginNamespace(mPluginNamespace);
    return plugin;
}

void Slice::setPluginNamespace(const char* pluginNamespace)
{
    //set library namespace
    mPluginNamespace = pluginNamespace;
}

const char* Slice::getPluginNamespace() const
{
    //get library namespace
    return mPluginNamespace;
}

nvinfer1::DataType Slice::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
   //output的类型 目前看只有一种类型
    assert(inputTypes[0] == DataType::kFLOAT || inputTypes[0] == DataType::kHALF);
    return inputTypes[1];
}

bool Slice::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    std::cout<<"isOutputBroadcastAcrossBatch"<<outputIndex<<" "<<inputIsBroadcasted[outputIndex]<<" "<<nbInputs<<std::endl;
    // if(outputIndex==0)
    // return false;
    // else 
    // return true;
    return true;
}

bool Slice::canBroadcastInputAcrossBatch(int inputIndex) const
{
    // if(inputIndex)
    //     return true;
    // else
    // {
    //     return false;
    // }
    std::cout<<"canBroadcastInputAcrossBatch  "<<inputIndex<<std::endl;

    return true;
    
}
void Slice::attachToContext(//无
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{

}

// Detach the plugin object from its execution context.
void Slice::detachFromContext() {}
SliceCreator::SliceCreator()
{
    mPluginAttributes.emplace_back(PluginField("x1", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("y1", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("z1", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("x2", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("y2", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("z2", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("z2", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}
const char* SliceCreator::getPluginName() const
{
    //Plugin Name
    return SLICE_PLUGIN_NAME;
}

// Returns the plugin version
const char* SliceCreator::getPluginVersion() const
{
    //Plugin Version
    return SLICE_PLUGIN_VERSION;
}

// Returns the plugin field names
const nvinfer1::PluginFieldCollection* SliceCreator::getFieldNames()
{
    //plugin的参数列表
    return &mFC;
}
nvinfer1::IPluginV2IOExt* SliceCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    //收集参数 使用属性时需要安装顺序插入
    const PluginField* fields = fc->fields;
    // Default init values for TF SSD network
    int x1,x2,y1,y2,z1,z2;
    nvinfer1::DataType type;
    // Read configurations from  each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "x1"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            x1 = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "y1"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            y1 = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "z1"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            z1 = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "x2"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            x2 = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "y2"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            y2 = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "z2"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            z2 = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "type"))
        {
            switch (fields[i].type)
            {
                case PluginFieldType::kFLOAT32: type=nvinfer1::DataType::kFLOAT;break;
                case PluginFieldType::kFLOAT16: type=nvinfer1::DataType::kHALF;break;
            }


        }
        
    }
    auto* obj = new Slice(x1,y1,z1,x2,y2,z2,type);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}
nvinfer1::IPluginV2IOExt* SliceCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    // 推理中使用plugin
    auto* obj = new Slice(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}