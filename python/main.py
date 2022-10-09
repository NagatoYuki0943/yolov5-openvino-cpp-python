"""openvino图片需要调整BGR2RGB, 并调整通道为[B, C, H, W], 且需要归一化, 这些操作可以通过模型指定
"""


from pathlib import Path

import openvino.runtime as ov
from openvino.preprocess import PrePostProcessor
from openvino.preprocess import ColorFormat
from openvino.runtime import Layout, Type

import numpy as np
import cv2

import time

import os, sys
os.chdir(sys.path[0])


MODEL_PATH = "../model/yolov5n.xml"
IMAGE_PATH = "../imgs/bus.jpg"
SCORE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
CONFIDENCE_THRESHOLD = 0.4


def resize_and_pad(image, new_shape):
    """缩放图片并填充为正方形

    Args:
        image (np.Array): 图片
        new_shape (Tuple): [h, w]

    Returns:
        Tuple: 所放的图片, 填充的宽, 填充的高
    """
    old_size = image.shape[:2]
    ratio = float(new_shape[-1]/max(old_size)) #fix to accept also rectangular images
    new_size = tuple([int(x*ratio) for x in old_size])
    # 缩放高宽的长边为640
    image = cv2.resize(image, (new_size[1], new_size[0]))
    # 查看高宽距离640的长度
    delta_w = new_shape[1] - new_size[1]
    delta_h = new_shape[0] - new_size[0]
    # 使用灰色填充到640*640的形状
    color = [100, 100, 100]
    img_reized = cv2.copyMakeBorder(image, 0, delta_h, 0, delta_w, cv2.BORDER_CONSTANT, value=color)

    return img_reized, delta_w ,delta_h


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


def get_model(model_path):
    """获取模型

    Args:
        model_path (str): 模型路径

    Returns:
        CompileModel: 编译好的模型
    """
    # Step 1. Initialize OpenVINO Runtime core
    core = ov.Core()
    # Step 2. Read a model
    model = core.read_model(str(Path(model_path)))

    # Step 4. Inizialize Preprocessing for the model
    ppp = PrePostProcessor(model)
    # Specify input image format
    ppp.input().tensor().set_element_type(Type.u8).set_layout(Layout("NHWC")).set_color_format(ColorFormat.BGR)
    #  Specify preprocess pipeline to input image without resizing
    ppp.input().preprocess().convert_element_type(Type.f32).convert_color(ColorFormat.RGB).scale([255., 255., 255.])
    # Specify model's input layout
    ppp.input().model().set_layout(Layout("NCHW"))
    #  Specify output results format
    ppp.output().tensor().set_element_type(Type.f32)
    # Embed above steps in the graph
    model = ppp.build()
    compiled_model = core.compile_model(model, "CPU")

    return compiled_model


def post(detections, delta_w ,delta_h, img):
    """后处理

    Args:
        detections (np.Array): 检测到的数据 [25200, 85]
        delta_w (int):  填充的宽
        delta_h (int):  填充的高
        img (np.Array): 原图

    Returns:
        np.Array: 绘制好的图片
    """
    boxes = []
    class_ids = []
    confidences = []
    for prediction in detections:
        confidence = prediction[4].item()
        if confidence >= CONFIDENCE_THRESHOLD:
            classes_scores = prediction[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > .25):
                confidences.append(confidence)
                class_ids.append(class_id)
                # 不是0~1之间的数据
                x, y, w, h = prediction[0].item(), prediction[1].item(), prediction[2].item(), prediction[3].item()
                xmin = x - (w / 2)
                ymin = y - (h / 2)
                box = np.array([xmin, ymin, w, h])
                boxes.append(box)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD)

    detections = []
    for i in indexes:
        j = i.item()
        detections.append({"class_index": class_ids[j], "confidence": confidences[j], "box": boxes[j]})

    # Step 9. Print results and save Figure with detections
    for detection in detections:

        box = detection["box"]
        classId = detection["class_index"]
        confidence = detection["confidence"]
        print( f"Bbox {i} Class: {classId} Confidence: {confidence}, Scaled coords: [ cx: {(box[0] + (box[2] / 2)) / img.shape[1]}, cy: {(box[1] + (box[3] / 2)) / img.shape[0]}, w: {box[2]/ img.shape[1]}, h: {box[3] / img.shape[0]} ]" )

        # 还原到原图尺寸
        box[0] = box[0] / ((640-delta_w) / img.shape[1])
        box[2] = box[2] / ((640-delta_w) / img.shape[1])
        box[1] = box[1] / ((640-delta_h) / img.shape[0])
        box[3] = box[3] / ((640-delta_h) / img.shape[0])

        xmax = box[0] + box[2]
        ymax = box[1] + box[3]
        img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(xmax), int(ymax)), (0, 255, 0), 3)
        img = cv2.rectangle(img, (int(box[0]), int(box[1]) - 20), (int(xmax), int(box[1])), (0, 255, 0), cv2.FILLED)
        img = cv2.putText(img, str(classId) + " " + "{:.2f}".format(confidence),
                          (int(box[0]), int(box[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    return img


def main():
    # 获取图片,扩展的宽高
    img, input_tensor, delta_w ,delta_h = get_image(IMAGE_PATH)

    # 获取模型
    compiled_model = get_model(MODEL_PATH)

    start = time.time()
    # Step 6. Create an infer request for model inference
    infer_request = compiled_model.create_infer_request()
    infer_request.infer({0: input_tensor})

    # Step 7. Retrieve inference results
    output = infer_request.get_output_tensor()
    detections = output.data[0]

    # Step 8. Postprocessing including NMS
    img = post(detections, delta_w ,delta_h, img)
    end = time.time()
    print((end - start) * 1000)

    cv2.imwrite("./openvion_det.png", img)


if __name__ == '__main__':
    main()