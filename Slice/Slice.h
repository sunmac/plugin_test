#include "plugin.h"
#include<vector>
#include<NvInferRuntimeCommon.h>
#include"kernel.h"

using namespace nvinfer1::plugin;

namespace nvinfer1
{
    namespace plugin
    {
        class Slice : public IPluginV2IOExt
        {
            public:
                Slice(int x1,int y1,int z1,int x2,int y2,int z2,nvinfer1::DataType inputtpye);
                
                Slice(const void* data, size_t length);
                
                int getNbOutputs() const override;
                
                Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;
                
                int initialize() override;

                void terminate() override;
                
                size_t getWorkspaceSize(int maxBatchSize) const override;

                int enqueue(
                    int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;
                
                size_t getSerializationSize() const override;

                void serialize(void* buffer) const override;

                void configurePlugin(const PluginTensorDesc * in,int nbInput,const PluginTensorDesc * out,int nbOutput )override;

                // bool supportsFormat(DataType type, PluginFormat format) const override;

                bool supportsFormatCombination	(int pos,const PluginTensorDesc* inOut,int nbInputs,int nbOutputs)	const override;

                const char* getPluginType() const override;

                const char* getPluginVersion() const override;
                
                void destroy() override;

                IPluginV2IOExt* clone() const override;

                void setPluginNamespace(const char* pluginNamespace) override;

                nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputType, int nbInputs) const override;

                const char* getPluginNamespace() const override;

                bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;

                bool canBroadcastInputAcrossBatch(int inputIndex) const override;

                void setClipParam(bool clip);
                
                void attachToContext(
                    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) override;

                void detachFromContext() override;
            private:
            int x1,x2,y1,y2,z1,z2;
            const char* mPluginNamespace;
            nvinfer1::DataType mType;
        };

        class SliceCreator:public BaseCreator
        {
            public:
                SliceCreator();
                ~SliceCreator() override=default;
                const char* getPluginName() const override;
                const char* getPluginVersion() const override;
                const PluginFieldCollection* getFieldNames() override;
                IPluginV2IOExt* createPlugin(const char* name, const PluginFieldCollection* slice) override;
                IPluginV2IOExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;
            private:
                static PluginFieldCollection mFC;

                // Parameters for DetectionOutput
                //DetectionOutputParameters params;
                static std::vector<PluginField> mPluginAttributes;
        };
        REGISTER_TENSORRT_PLUGIN(SliceCreator);

    }
}