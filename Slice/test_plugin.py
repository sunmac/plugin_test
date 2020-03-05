import tensorrt as trt
import numpy as np
from PIL import Image
#import pycuda.driver as cuda
#import pycuda.autoinit
import time
import ctypes
import common
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

#ctypes.CDLL("/code/OCRTransformer/plugin/Slice/libOCRTrt.so")
trt.init_libnvinfer_plugins(TRT_LOGGER, '')
PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list
#c=[print(plugin_creator.name) for plugin_creator in PLUGIN_CREATORS]
x1,y1,z1=1,8,1
x2,y2,z2=1,8,1
def init_construct_network():
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network:
        builder.max_workspace_size = 1024*1024*1024
        # builder.fp16_mode =True
        matrix1 = network.add_input(name="matrix1", dtype=trt.float32, shape=(x1,y1,z1))
        matrix2 = network.add_input(name="matrix2", dtype=trt.float32, shape=(x2,y2,z2))
        sum_m1m2=network.add_elementwise(matrix1,matrix2,trt.ElementWiseOperation.SUM)


        slice_layer=network.add_plugin_v2(inputs=[sum_m1m2.get_output(0),sum_m1m2.get_output(0)], plugin=get_trt_slice("LICE_TRT",2))
        # slice_layer=network.add_plugin_v2(inputs=[sum_m1m2,sum_m1m2], plugin=get_trt_slice("LICE_TRT",7))

        # slice_layer=network.add_plugin_v2(inputs=[matrix1,matrix2], plugin=get_trt_slice("LICE_TRT",7))

        # network.mark_output(sum_m1m2.get_output(0))
        network.mark_output(slice_layer.get_output(0))

        return builder.build_cuda_engine(network)
def get_trt_slice(plugin_name,input_tensor):
    for plugin_creator in PLUGIN_CREATORS:
            if plugin_creator.name == plugin_name:
                x1_= trt.PluginField("x1", np.array([x1], dtype=np.int32), trt.PluginFieldType.INT32)
                y1_= trt.PluginField("y1", np.array([y1], dtype=np.int32), trt.PluginFieldType.INT32)
                z1_ = trt.PluginField("z1", np.array([z1], dtype=np.int32), trt.PluginFieldType.INT32)
                x2_ = trt.PluginField("x2", np.array([x2], dtype=np.int32), trt.PluginFieldType.INT32)
                y2_ = trt.PluginField("y2", np.array([y2], dtype=np.int32), trt.PluginFieldType.INT32)
                z2_ = trt.PluginField("z2", np.array([z2], dtype=np.int32), trt.PluginFieldType.INT32)
                z2_ = trt.PluginField("type", np.array([0], dtype=np.float32), trt.PluginFieldType.FLOAT32)

                field_collection = trt.PluginFieldCollection([x1_, y1_, z1_, x2_, y2_,z2_])         
                plugin = plugin_creator.create_plugin(name=plugin_name, field_collection=field_collection)
                return plugin
def main():
    
    engine = init_construct_network()
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)

    with engine.create_execution_context() as context:
        print(np.ones((x1*y1*z1),np.float32).reshape(-1))
        np.copyto(inputs[0].host,np.ones((x1*y1*z1),np.float32).reshape(-1))
        np.copyto(inputs[1].host,np.ones((x2*y2*z2),np.float32).reshape(-1))

        time_start=time.time()
        output = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        time_end=time.time()
        print("time ",time_end-time_start)
        print(output[0].reshape((x1*y1*z1)))
        data=output[0].reshape((x1*y1*z1))
        print(data[0])
        print(output[1].reshape((x1*y1*z1)))


        
        

    print("ok")

if __name__ == "__main__":
    main()