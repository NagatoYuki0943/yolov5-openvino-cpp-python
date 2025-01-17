#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <string>

using namespace std;


const float CONFIDENCE_THRESHOLD = 0.25;
const float SCORE_THRESHOLD = 0.2;
const float NMS_THRESHOLD = 0.45;


/**
 * save classid confidence and box
 */
struct Detection
{
    int class_id{};
    float confidence{};
    cv::Rect box;
};


/**
 * origin image, resized image and h w
 */
struct Resize
{
    cv::Mat img;
    cv::Mat resized_image;
    int dw{};
    int dh{};
};


/**
 * Zooms the picture to the specified size and fills the edges
 * @param img
 * @param new_shape
 * @return resize
 */
Resize resize_and_pad(cv::Mat& img, cv::Size new_shape) {
    float width = img.cols;
    float height = img.rows;
    auto r = float(new_shape.width / max(width, height));
    int new_unpadW = int(round(width * r));
    int new_unpadH = int(round(height * r));
    Resize resize;
    resize.img = img;
    cv::resize(img, resize.resized_image, cv::Size(new_unpadW, new_unpadH), 0, 0, cv::INTER_AREA);

    resize.dw = new_shape.width - new_unpadW;
    resize.dh = new_shape.height - new_unpadH;
    cv::Scalar color = cv::Scalar(100, 100, 100);
    cv::copyMakeBorder(resize.resized_image, resize.resized_image, 0, resize.dh, 0, resize.dw, cv::BORDER_CONSTANT, color);

    return resize;
}


/**
 * get image and resized image
 * @param image_path
 * @return Resize
 */
Resize get_image(const string& image_path){
    // Step 3. Read input image
    cv::Mat img = cv::imread(image_path);
    // resize image
    Resize res = resize_and_pad(img, cv::Size(640, 640));
    return res;
}


/**
 * input(0)/output(0) 按照id找指定的输入输出，不指定找全部的输入输出
 *
 *  input().tensor()       有7个方法
 *  ppp.input().tensor().set_color_format().set_element_type().set_layout()
 *                      .set_memory_type().set_shape().set_spatial_dynamic_shape().set_spatial_static_shape();
 *
 *  output().tensor()      有2个方法
 *  ppp.output().tensor().set_layout().set_element_type();
 *
 *  input().preprocess()   有8个方法
 *  ppp.input().preprocess().convert_color().convert_element_type().mean().scale()
 *                          .convert_layout().reverse_channels().resize().custom();
 *
 *  output().postprocess() 有3个方法
 *  ppp.output().postprocess().convert_element_type().convert_layout().custom();
 *
 *  input().model()  只有1个方法
 *  ppp.input().model().set_layout();
 *
 *  output().model() 只有1个方法
 *  ppp.output().model().set_layout();
 **/


/**
 * get openvino model
 * @param model_path
 * @return CompiledModel
 */
ov::CompiledModel get_model(const string& model_path, const string& device="CPU"){
    // Step 1. Initialize OpenVINO Runtime core
    ov::Core core;
    // Step 2. Read a model
    std::shared_ptr<ov::Model> model = core.read_model(model_path);

    // Step 4. Inizialize Preprocessing for the model
    // https://mp.weixin.qq.com/s/4lkDJC95at2tK_Zd62aJxw
    // https://blog.csdn.net/sandmangu/article/details/107181289
    // https://docs.openvino.ai/latest/openvino_2_0_preprocessing.html
    ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);

    // Specify input image format   input(0) refers to the 0th input.
    ppp.input(0).tensor()
        .set_color_format(ov::preprocess::ColorFormat::BGR)
        .set_element_type(ov::element::u8)
        .set_layout(ov::Layout("NHWC"));

    // Specify preprocess pipeline to input image without resizing
    ppp.input(0).preprocess()
        .convert_color(ov::preprocess::ColorFormat::RGB)
        .convert_element_type(ov::element::f32)
        .scale({255., 255., 255.});

    //  Specify model's input layout
    ppp.input(0).model().set_layout(ov::Layout("NCHW"));

    // Specify output results format
    ppp.output(0).tensor().set_element_type(ov::element::f32);

    // Embed above steps in the graph
    model = ppp.build();
    ov::CompiledModel compiled_model = core.compile_model(model, device);
    return compiled_model;
}


/**
 * Post processing
 * @param detections
 * @param output_shape
 * @param res Resize
 */
void post(float *detections, ov::Shape output_shape, Resize& res){
    // Step 8. Postprocessing including NMS
    std::vector<cv::Rect> boxes;
    vector<int> class_ids;
    vector<float> confidences;

    for (int i = 0; i < output_shape[1]; i++){
        float *detection = &detections[i * output_shape[2]];

        float confidence = detection[4];
        if (confidence >= CONFIDENCE_THRESHOLD){
            float *classes_scores = &detection[5];
            cv::Mat scores(1, output_shape[2] - 5, CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

            if (max_class_score > SCORE_THRESHOLD){

                confidences.push_back(confidence);

                class_ids.push_back(class_id.x);

                float x = detection[0];
                float y = detection[1];
                float w = detection[2];
                float h = detection[3];

                float xmin = x - (w / 2);
                float ymin = y - (h / 2);

                boxes.push_back(cv::Rect(xmin, ymin, w, h));
            }
        }
    }
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
    std::vector<Detection> output;
    for (int i = 0; i < nms_result.size(); i++)
    {
        Detection result;
        int idx = nms_result[i];
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result);
    }


    // Step 9. Print results and save Figure with detections
    for (int i = 0; i < output.size(); i++)
    {
        auto detection = output[i];
        auto box = detection.box;
        auto classId = detection.class_id;
        auto confidence = detection.confidence;
        // resize to origin size
        float rx = (float)res.img.cols / (float)(res.resized_image.cols - res.dw);
        float ry = (float)res.img.rows / (float)(res.resized_image.rows - res.dh);
        box.x = rx * box.x;
        box.y = ry * box.y;
        box.width = rx * box.width;
        box.height = ry * box.height;
        cout << "Bbox" << i + 1 << ": Class: " << classId << " "
             << "Confidence: " << confidence << " Scaled coords: [ "
             << "cx: " << (float)(box.x + (box.width / 2)) / res.img.cols << ", "
             << "cy: " << (float)(box.y + (box.height / 2)) / res.img.rows << ", "
             << "w: " << (float)box.width / res.img.cols << ", "
             << "h: " << (float)box.height / res.img.rows << " ]" << endl;
        float xmax = box.x + box.width;
        float ymax = box.y + box.height;
        cv::rectangle(res.img, cv::Point(box.x, box.y), cv::Point(xmax, ymax),
                      cv::Scalar(0, 255, 0), 3);
        cv::rectangle(res.img, cv::Point(box.x, box.y - 20), cv::Point(xmax, box.y),
                      cv::Scalar(0, 255, 0), cv::LineTypes::FILLED);
        string text = std::to_string(classId) + " " + std::to_string(confidence).substr(0, 4);
        cv::putText(res.img, text, cv::Point(box.x, box.y - 5), cv::HersheyFonts::FONT_HERSHEY_SIMPLEX,
                    0.5, cv::Scalar(0, 0, 0));
    }
}

int main(){
    //                                                                       or  yolov5s_openvino_model
    string model_path = "D:/ai/code/ultralytics/yolov5-openvino-cpp-python/weights/yolov5s_openvino_model_quantization/yolov5s.xml";
    string image_path = "D:/ai/code/ultralytics/yolov5-openvino-cpp-python/imgs/bus.jpg";

    // Step 1. Get images
    Resize res = get_image(image_path);

    // Step 2. get CompiledModel
    ov::CompiledModel compiled_model = get_model(model_path, "CPU");
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    // time
    auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    // Step 3. Create tensor from image
    auto *input_data = (float *) res.resized_image.data;
    ov::Tensor input_tensor = ov::Tensor(compiled_model.input().get_element_type(), compiled_model.input().get_shape(), input_data);

    // Step 4. Create an infer request for model inference
    // many ways to infer
    // https://docs.openvino.ai/latest/openvino_2_0_inference_pipeline.html
    // https://docs.openvino.ai/latest/notebooks/002-openvino-api-with-output.html#
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();

    // Step 7. Retrieve inference results
    const ov::Tensor &output_tensor = infer_request.get_output_tensor(0);
    ov::Shape output_shape = output_tensor.get_shape();
    auto *detections = output_tensor.data<float>();

    // Step 8. Post processing
    post(detections, output_shape, res);

    // time
    auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    cout << end - start << "ms" << endl;
    // save image
    cv::imwrite("./openvino_detection.png", res.img);
    cv::imshow("openvino_detection", res.img);
    cv::waitKey(0);
    return 0;
}
