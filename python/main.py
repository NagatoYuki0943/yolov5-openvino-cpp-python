"""openvino图片需要调整BGR2RGB, 并调整通道为[B, C, H, W], 且需要归一化, 这些操作可以通过模型指定
"""


from pathlib import Path

import openvino.runtime as ov
from openvino.preprocess import PrePostProcessor
from openvino.preprocess import ColorFormat
from openvino.runtime import Layout, Type

import numpy as np
import cv2

from utils import resize_and_pad, post, get_index2label

import time

import os, sys
os.chdir(sys.path[0])


CONFIDENCE_THRESHOLD = 0.25
SCORE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.45


def get_image(image_path):
    """获取图像

    Args:
        image_path (str): 图片路径

    Returns:
        Tuple: 原图, 输入的tensor, 填充的宽, 填充的高
    """
    img = cv2.imread(str(Path(image_path)))
    # img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)    # BGR2RGB                  ppp实现了

    img_reized, delta_w ,delta_h = resize_and_pad(img, (640, 640))

    img_reized = img_reized.astype(np.float32)
    # input_tensor /= 255.0                                 # 归一化                    ppp实现了

    # img_reized = img_reized.transpose(2, 0, 1)            # [H, W, C] -> [C, H, W]    ppp实现了
    input_tensor = np.expand_dims(img_reized, 0)            # [C, H, W] -> [B, C, H, W]

    return img, input_tensor, delta_w ,delta_h


"""openvino图片预处理方法
input(0)/output(0) 按照id找指定的输入输出,不指定找全部的输入输出

# input().tensor()       有7个方法
ppp.input().tensor().set_color_format().set_element_type().set_layout() \
                    .set_memory_type().set_shape().set_spatial_dynamic_shape().set_spatial_static_shape()

# output().tensor()      有2个方法
ppp.output().tensor().set_layout().set_element_type()

# input().preprocess()   有8个方法
ppp.input().preprocess().convert_color().convert_element_type().mean().scale() \
                        .convert_layout().reverse_channels().resize().custom()

# output().postprocess() 有3个方法
ppp.output().postprocess().convert_element_type().convert_layout().custom()

# input().model()  只有1个方法
ppp.input().model().set_layout()

# output().model() 只有1个方法
ppp.output().model().set_layout()
"""


def get_model(model_path, device='CPU'):
    """获取模型

    Args:
        model_path (str): 模型路径
        device (str):     模型设备, CPU or GPU
    Returns:
        CompileModel: 编译好的模型
    """
    # Step 1. Initialize OpenVINO Runtime core
    core = ov.Core()
    # Step 2. Read a model
    model = core.read_model(str(Path(model_path)))

    # Step 4. Inizialize Preprocessing for the model  openvino数据预处理
    # https://mp.weixin.qq.com/s/4lkDJC95at2tK_Zd62aJxw
    # https://blog.csdn.net/sandmangu/article/details/107181289
    # https://docs.openvino.ai/latest/openvino_2_0_preprocessing.html
    ppp = PrePostProcessor(model)
    # 设定图片数据类型，形状，通道排布为BGR
    ppp.input(0).tensor().set_color_format(ColorFormat.BGR).set_element_type(Type.u8).set_layout(Layout("NHWC"))
    # 预处理: 改变类型,转换为RGB,通道归一化(标准化中的除以均值也能这样求),还有.mean()均值 mean要在scale前面
    ppp.input(0).preprocess().convert_color(ColorFormat.RGB).convert_element_type(Type.f32).scale([255., 255., 255.])
    # 指定模型输入形状
    ppp.input(0).model().set_layout(Layout("NCHW"))
    # 指定模型输出类型
    ppp.output().tensor().set_element_type(Type.f32)
    # Embed above steps in the graph
    model = ppp.build()
    compiled_model = core.compile_model(model, device)

    return compiled_model


def main():
    #                        yolov5s_openvino_model_quantization
    MODEL_PATH = "../weights/yolov5s_openvino_model/yolov5s.xml"
    IMAGE_PATH = "../imgs/bus.jpg"
    YAML_PATH  = "../weights/yolov5s_openvino_model/yolov5s.yaml"

    # 1.获取图片,扩展的宽高
    img, input_tensor, delta_w ,delta_h = get_image(IMAGE_PATH)

    # 2.获取模型
    compiled_model = get_model(MODEL_PATH, 'CPU')

    # 3.获取label
    index2label = get_index2label(YAML_PATH)

    # 4.获取模型的一些数据,多个输出和输出使用 [0] [1] 取出
    inputs = compiled_model.inputs
    outputs = compiled_model.outputs
    print(f"inputs: {inputs}")                                      # inputs: [<ConstOutput: names[images] shape{1,640,640,3} type: u8>]
    print(inputs[0].index, inputs[0].names, inputs[0].shape, )      # 0 {'images'} {1, 640, 640, 3}
    print(f"outputs: {outputs}")                                    # outputs: [<ConstOutput: names[output0] shape{1,25200,85} type: f32>]
    print(outputs[0].index, outputs[0].names, outputs[0].shape, )   # 0 {'output0'} {1, 25200, 85}

    start = time.time()

    # 5.推理 多种方式
    # https://docs.openvino.ai/latest/openvino_2_0_inference_pipeline.html
    # https://docs.openvino.ai/latest/notebooks/002-openvino-api-with-output.html#

    # 5.1 使用推理请求
    # infer_request = compiled_model.create_infer_request()
    # results       = infer_request.infer({inputs[0]: input_tensor})    # 直接返回推理结果
    # results       = infer_request.infer({0: input_tensor})            # 直接返回推理结果
    # results       = infer_request.infer([input_tensor])               # 直接返回推理结果

    # 5.2 模型直接推理
    # results       = compiled_model({inputs[0]: input_tensor})
    # results       = compiled_model({0: input_tensor})
    results = compiled_model([input_tensor])
    # print(outputs.keys)           # <built-in method keys of dict object at 0x0000019A7C7C68C0>

    # 获取输出
    # result = infer_request.get_output_tensor(outputs[0].index)    # outputs[0].index 可以用0 1代替
    result = results[outputs[0]]
    print(result.shape)             # (1, 25200, 85)
    detections = result[0]          # 去除batch

    # Step 8. Postprocessing including NMS
    img = post(detections, delta_w ,delta_h, img, CONFIDENCE_THRESHOLD, SCORE_THRESHOLD, NMS_THRESHOLD, index2label)
    end = time.time()
    print('time:', (end - start) * 1000)

    cv2.imwrite("./openvion_det.png", img)


if __name__ == '__main__':
    main()
