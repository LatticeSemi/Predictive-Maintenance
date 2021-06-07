import tensorflow as tf
import numpy as np
import cv2
import sys
import glob
from tensorflow.python.platform import gfile
import argparse

ModelConfig = {
    "input_arrays": "random_shuffle_queue_DequeueMany",
    "input_shape": "1,64,64,1",
    "output_arrays": "logit/BiasAdd",
    "optimization": True,
    "quantization": True
}

representative_data_path = None


def representative_dataset_gen():
    for f in glob.glob(representative_data_path + "/*"):
        im = cv2.imread(f)
        if im is None:
            print("File read failed: {}".format(f))
            sys.exit(0)
        # else:
        #     print("File read {}".format(f))
        im = im.astype(np.float32, copy=False)
        resolution = list(map(int, ModelConfig['input_shape'].split(',')))[1]
        # print("resolution={}".format(resolution))
        im = cv2.resize(im, (resolution, resolution), interpolation=cv2.INTER_AREA)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # im = im/128.0 # This was used for sensAI as it uses [0,2] as the blob range
        im = im - 128.0  # TFLITE (INT8) needs this since it uses [-128,127] as the blob range
        im = np.expand_dims(im, axis=0)
        im = np.expand_dims(im, axis=3)
        yield [im]


def main(args):
    global representative_data_path
    representative_data_path = args.input_path
    with tf.compat.v1.Session() as sess:
        # print(args.pb)
        with gfile.FastGFile(args.pb, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            g_in = tf.import_graph_def(graph_def)

            converter = tf.compat.v1.lite.TFLiteConverter(
                graph_def=graph_def,
                input_tensors=None,
                output_tensors=None,
                input_arrays_with_shape=[(ModelConfig['input_arrays'],
                                          list(map(int, ModelConfig['input_shape'].split(','))))],
                output_arrays=[ModelConfig['output_arrays']]
            )

            if ModelConfig['optimization']:  # This includes quantization too
                converter.optimizations = [tf.compat.v1.lite.Optimize.DEFAULT]  # 8b int dynamic quant
                # converter.optimizations = [tf.compat.v1.lite.Optimize.OPTIMIZE_FOR_SIZE]

            # Mean and Std statistical information for optimization/quantization
            if ModelConfig['quantization']:
                converter.representative_dataset = representative_dataset_gen
                converter.target_spec.supported_ops = [
                    tf.compat.v1.lite.OpsSet.TFLITE_BUILTINS_INT8]  # INT8 based build in operations
                # converter.target_spec.supported_ops = [
                # tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
                # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS] #
                # All built in operations

                converter.inference_input_type = tf.int8  # Unsigned 8b #compat.v1.lite.constants.UINT8
                converter.inference_output_type = tf.int8  # Signed   8b #compat.v1.lite.constants.INT8
                converter.inference_type = tf.float32
            else:
                converter.target_spec.supported_types = [tf.float16]

            tflite_qaunt_model = converter.convert()
            open(args.tflite_path, "wb").write(tflite_qaunt_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default='./data/Dataset/dataset/0',
                        help="Typical input image directory")
    parser.add_argument("--tflite_path", default='./motor-model.tflite', help="TFLite output file")
    parser.add_argument("--pb", default='logs/train/model_frozenforInference.pb', help="Frozen .pb file")
    args = parser.parse_args()
    main(args)
