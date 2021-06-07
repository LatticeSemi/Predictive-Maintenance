import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tflite_path", type=str, required=True, help="TensorFlowLite Model Path")
    parser.add_argument("--output_file", type=str, required=True, help="CC model path")

    args = parser.parse_args()

    cmd = 'xxd -i '+ args.tflite_path + ' > ' + args.output_file

    ret = os.system(cmd)
    if ret == 0:
        print("Model compiled successfully")
    else:
       print("Model compilation failed")
