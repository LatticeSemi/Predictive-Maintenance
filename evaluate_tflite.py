import tensorflow as tf
import numpy as np
import glob
import cv2
import sys
import os
import argparse


# Helper function to run inference on a TFLite model
def run_tflite_model(quant=True, test_label=0, tflite_file="motor-model.tflite",
                     input_data_path="data/Dataset/dataset"):
    #   global test_images

    # Initialize the interpreter
    interpreter = tf.lite.Interpreter(model_path=tflite_file)
    # allocate the tensors
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # print(input_details)
    # print(output_details)
    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    gray = input_shape[-1] == 1
    # print("gray: {},quant: {},test_label: {}".format(gray, quant, test_label))
    # ip_scale,ip_zero = input_details[0]['quantization_parameters']["scales"],input_details[0][
    # 'quantization_parameters']["zero_points"] op_scale,op_zero = output_details[0]['quantization_parameters'][
    # "scales"],output_details[0]['quantization_parameters']["zero_points"]
    outputs = []

    total_test_imgs = 0
    for f in glob.glob(input_data_path + "/*"):
        im = cv2.imread(f)
        if im is None:
            print("File read failed: {}".format(f))
            sys.exit(0)
        resolution = input_shape[1]
        im = cv2.resize(im, (resolution, resolution), interpolation=cv2.INTER_AREA)
        if gray:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = im - 128.0  # TFLITE (INT8) needs this since it uses [-128,127] as the blob range
        if quant:
            im = im.astype(np.int8, copy=False)
        else:
            im = im.astype(np.float32, copy=False)
        im = np.expand_dims(im, axis=0)
        if gray:
            im = np.expand_dims(im, axis=3)
        interpreter.set_tensor(input_details[0]['index'], im)

        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        outputs.append(interpreter.get_tensor(output_details[0]['index']).argmax() == test_label)
        if interpreter.get_tensor(output_details[0]['index']).argmax() != test_label:
            print("Wrong prediction for image :")
            print(interpreter.get_tensor(output_details[0]['index']).argmax(), test_label)
        total_test_imgs += 1

    accuracy = (np.sum(outputs) * 100) / total_test_imgs
    return [accuracy, total_test_imgs, np.sum(outputs)]


def main(args):
    input_data_path_normal = os.path.join(args.dataset_path, "1")
    input_data_path_broken = os.path.join(args.dataset_path, "0")
    acc_tl_norm, total_images_tl_norm, correct_out_tl_norm = run_tflite_model(quant=True, test_label=1,
                                                                              input_data_path=input_data_path_normal)

    acc_tl_brok, total_images_tl_brok, correct_out_tl_brok = run_tflite_model(quant=True, test_label=0,
                                                                              input_data_path=input_data_path_broken)
    print("Broken Accuracy {}\nNormal Accuracy {}".format(acc_tl_brok, acc_tl_norm))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="dataset directory")
    parser.add_argument("--tflite_path", required=True, help="Freeze graph for inference.")
    args = parser.parse_args()
    main(args)
