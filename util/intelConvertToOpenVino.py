import openvino as ov

# Path to your ONNX model
onnx_model_path = 'nextMoveSnake.onnx'

# Convert the ONNX model to OpenVINO IR format
ir_model = ov.convert_model(onnx_model_path)

# Save the converted model to the current directory with proper file extensions
output_model_path_xml = 'nextMoveSnake.xml'

ov.save_model(ir_model, output_model_path_xml)

print("Done")