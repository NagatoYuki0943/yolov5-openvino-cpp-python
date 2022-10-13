# 安装openvino

## 下载安装

> https://www.intel.cn/content/www/cn/zh/developer/tools/openvino-toolkit/download.html
>
> 下载2022.1版本，最新版本没提供对应的opencv
>
> 勾选C++
>
> 下载完成后安装

## 下载对应的opencv

> 在`openvino_2022.1.0.643\extras\scripts\`文件夹下有`download_opencv.ps1`
>
> 运行会将opencv下载到`openvino_2022.1.0.643\extras\`目录下
>
> 可以运行opencv目录下的`ffmpeg-download.ps1`下载ffmpeg

## 配置环境变量

```yaml
#opencv
D:\ai\openvino\openvino_2022.1.0.643\extras\opencv\bin

#openvino
D:\ai\openvino\openvino_2022.1.0.643\runtime\bin\intel64\Debug
D:\ai\openvino\openvino_2022.1.0.643\runtime\bin\intel64\Release
D:\ai\openvino\openvino_2022.1.0.643\runtime\3rdparty\tbb\bin
```

# 错误

## 找不到`opencv_core_parallel_onetbb455_64d.dll`

> 没有问题，参考https://github.com/opencv/opencv/issues/20113
>
> debug模式下会显示，release不会显示

