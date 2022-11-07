# YOLOv5-Openvino-Cpp-Python

Example of performing inference with ultralytics [YOLOv5](https://github.com/ultralytics/yolov5) using the 2022.1.0 openvino API in C++ using Docker as well as python.

This repository is only for model inference using openvino. Therefore, it assumes the YOLOv5 model is already trained and exported to openvino (.bin, .xml) format. For further info check [YOLOv5](https://github.com/ultralytics/yolov5).

## YOLOv5-Openvino-Cpp

### Docker installation
This repository folder contains the Dockerfile to build a docker image with the Intel® Distribution of OpenVINO™ toolkit.

1) This command builds an image with OpenVINO™ 2022.1.0 release.
    ```
    docker build cpp -t openvino_container:2022.1.0
    ```
2) This command creates a docker container with OpenVINO™ 2022.1.0 release.
    ##### Windows
    ```
    docker run -it --rm -v %cd%:/yolov5-openvino openvino_container:2022.1.0
    ```
    ##### Linux/Mac
    ```
    docker run -it --rm -v $(pwd):/yolov5-openvino openvino_container:2022.1.0
    ```
### Cmake build

From within the docker run:
```
cd cpp && mkdir build && cd build
```
Then create the make file using cmake:
```
cmake -S ../ -O ./
```
Then compile the program using:
```
make
```
Then run the executable:
```
./main
```

## YOLOv5-Openvino-Python

### Usage

```
python -m venv /path/to/env

source /path/to/env/bin/activate # Linux/mac

\path\to\env\Script\activate # Windows
```
```
cd python
```
```
pip install -r requirements.txt
```

Then run the script:
```
python main.py
```

## Final Result:

![IMAGE_DESCRIPTION](./imgs/result.png)



# Error

### openvino DLL load failed while importing ie_api

> https://blog.csdn.net/qq_26815239/article/details/123047840
>
> 如果你使用的是 Python 3.8 或更高版本，并且是在Windows系统下通过pip安装的openvino，那么该错误的解决方案如下：

1. 进入目录 `your\env\site-packages\openvino\inference_engine`
2. 打开文件 `__init__.py`
3. 26行下添加一行

```python
        if os.path.isdir(lib_path):
            # On Windows, with Python >= 3.8, DLLs are no longer imported from the PATH.
            if (3, 8) <= sys.version_info:
                os.add_dll_directory(os.path.abspath(lib_path))
                os.environ['PATH'] = os.path.abspath(lib_path) + ';' + os.environ['PATH']	# 添加这一行
```
