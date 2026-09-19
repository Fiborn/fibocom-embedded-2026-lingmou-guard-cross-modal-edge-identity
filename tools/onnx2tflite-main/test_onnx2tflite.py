from onnx2tflite import onnx_converter
onnx_path = "adaface_ir18_webface4m.onnx"  # 需要转换的onnx文件位置
onnx_converter(
    onnx_model_path = onnx_path,
    need_simplify = True,
    output_path = "./",  # 输出的tflite存储路径
    target_formats = ['tflite'], # or ['keras'], ['keras', 'tflite']
    weight_quant = False,
    int8_model = False,
    int8_mean = None,
    int8_std = None,
    image_root = None
)
