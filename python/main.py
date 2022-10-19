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
    # img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)    # BGR2RGB                   ppp实现了

    img_reized, delta_w ,delta_h = resize_and_pad(img, (640, 640))

    img_reized = img_reized.astype(np.float32)
    # input_tensor /= 255.0                                 # 归一化                     ppp实现了

    # img_reized = img_reized.transpose(2, 0, 1)            # [H, W, C] -> [C, H, W]    ppp实现了
    input_tensor = np.expand_dims(img_reized, 0)            # [C, H, W] -> [B, C, H, W]

    return img, input_tensor, delta_w ,delta_h


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
    ppp = PrePostProcessor(model)
    # Specify input image format 设定图片数据类型，形状，通道排布为BGR
    ppp.input().tensor().set_element_type(Type.u8).set_layout(Layout("NHWC")).set_color_format(ColorFormat.BGR)
    #  Specify preprocess pipeline to input image without resizing 预处理：改变类型，转换为RGB，通道归一化
    ppp.input().preprocess().convert_element_type(Type.f32).convert_color(ColorFormat.RGB).scale([255., 255., 255.])
    # Specify model's input layout 指定模型输入形状
    ppp.input().model().set_layout(Layout("NCHW"))
    #  Specify output results format 指定模型输出类型
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

    # 获取图片,扩展的宽高
    img, input_tensor, delta_w ,delta_h = get_image(IMAGE_PATH)

    # 获取模型
    compiled_model = get_model(MODEL_PATH, 'CPU')

    # Step 6. Create an infer request for model inference
    infer_request = compiled_model.create_infer_request()

    # 获取模型的一些数据,多个输出和输出使用 [0] [1] 取出
    inputs_names = compiled_model.inputs
    outputs_names = compiled_model.outputs
    print(f"inputs_names: {inputs_names}")  # inputs_names: [<ConstOutput: names[images] shape{1,640,640,3} type: u8>]
    print(inputs_names[0].names, inputs_names[0].shape, inputs_names[0].index)      # {'images'} {1, 640, 640, 3} 0
    print(f"outputs_names: {outputs_names}")# outputs_names: [<ConstOutput: names[output0] shape{1,25200,85} type: f32>]
    print(outputs_names[0].names, outputs_names[0].shape, outputs_names[0].index)   # {'output0'} {1, 25200, 85} 0

    # 获取label
    index2label = get_index2label(YAML_PATH)

    start = time.time()
    # 设置输入
    # infer_request.infer({0: input_tensor}) # 两种方式
    infer_request.infer({inputs_names[0]: input_tensor})

    # 获取输出
    # output = infer_request.get_output_tensor()
    output = infer_request.get_output_tensor(outputs_names[0].index)
    detections = output.data[0]

    # Step 8. Postprocessing including NMS
    img = post(detections, delta_w ,delta_h, img, CONFIDENCE_THRESHOLD, SCORE_THRESHOLD, NMS_THRESHOLD, index2label)
    end = time.time()
    print((end - start) * 1000)

    cv2.imwrite("./openvion_det.png", img)


if __name__ == '__main__':
    main()
