#include "plugin.h"
#include<vector>
#include<NvInferRuntimeCommon.h>
#include"kernel.h"

using namespace nvinfer1::plugin;

namespace nvinfer1
{
    namespace plugin
    {
        class Slice : public IPluginV2DynamicExt 
        {
            public:
                Slice(int x1,int y1,int z1,int x2,int y2,int z2,nvinfer1::DataType inputtpye);
                
                Slice(const void* data, size_t length);
                
                int getNbOutputs() const override;
                
                DimsExprs  getOutputDimensions(int outputIndex, const DimsExprs *inputs, int nbInputs, IExprBuilder &exprBuilder)override;
                
                int initialize() override;

                void terminate() override;
                
                size_t getWorkspaceSize (const PluginTensorDesc *inputs, int nbInputs, const PluginTensorDesc *outputs, int nbOutputs)const override;

                int enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) override;
                
                size_t getSerializationSize() const override;

                void serialize(void* buffer) const override;

                void configurePlugin(const DynamicPluginTensorDesc *in, int nbInputs, const DynamicPluginTensorDesc *out, int nbOutputs)override;

                // bool supportsFormat(DataType type, PluginFormat format) const override;

                bool supportsFormatCombination	(int pos,const PluginTensorDesc* inOut,int nbInputs,int nbOutputs)	override;

                const char* getPluginType() const override;

                const char* getPluginVersion() const override;
                
                void destroy() override;

                IPluginV2DynamicExt* clone() const override;

                void setPluginNamespace(const char* pluginNamespace) override;

                nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputType, int nbInputs) const override;

                const char* getPluginNamespace() const override;



                void setClipParam(bool clip);
                
                // void attachToContext(
                //     cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) override;

                void detachFromContext() override;
            private:
            int x1,x2,y1,y2,z1,z2;
            const char* mPluginNamespace;
            nvinfer1::DataType mType;
            protected:
                // bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;
                // bool canBroadcastInputAcrossBatch(int inputIndex) const override;
                // size_t getWorkspaceSize(int maxBatchSize) const;
                using nvinfer1::IPluginV2DynamicExt::isOutputBroadcastAcrossBatch;
                using nvinfer1::IPluginV2DynamicExt::canBroadcastInputAcrossBatch;
        };

        class SliceCreator:public BaseCreator
        {
            public:
                SliceCreator();
                ~SliceCreator() override=default;
                const char* getPluginName() const override;
                const char* getPluginVersion() const override;
                const PluginFieldCollection* getFieldNames() override;
                IPluginV2* createPlugin(const char* name, const PluginFieldCollection* slice) override;
                IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;
            private:
                static PluginFieldCollection mFC;

                // Parameters for DetectionOutput
                //DetectionOutputParameters params;
                static std::vector<PluginField> mPluginAttributes;
        };
        REGISTER_TENSORRT_PLUGIN(SliceCreator);

    }
}