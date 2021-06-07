import os  # used for directory operations
import tensorflow as tf
import cv2
from PIL import Image  # used to read images from directory
import argparse
import sys
import random


def main(args):

	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)
	recordPath = os.path.abspath(args.output_dir)
	input_dataset_path = os.path.abspath(args.input_dataset)
	if not os.path.exists(input_dataset_path):
		print("Input dataset path doesn't exists !!")
		sys.exit(-1)

	# the best number of images stored in each tfrecord file
	print(args)
	bestNum = args.images_per_record
	num = 0
	recordFileNum = 0
	size = 64, 64

	# name format of the tfrecord files
	recordFileName = ("train.tfrecords-%.3d" % recordFileNum)
	print("Creating: ", recordFileName)
	# tfrecord file writer
	writer = tf.io.TFRecordWriter(os.path.join(recordPath, recordFileName))

	Image_List = []

	for class_name in os.listdir(input_dataset_path):
		
		for image in os.listdir(os.path.join(input_dataset_path, class_name)):
			img_path = os.path.join(input_dataset_path, class_name, image)
			# print(img_path)
			Image_List.append((img_path, class_name))

	for i in range(4):
		random.shuffle(Image_List)
			
	for (img_path,class_name) in Image_List:
		# print(img_path, int(class_name))
		img = cv2.imread(img_path)
		img = cv2.resize(img, size)
		img_raw = cv2.imencode('.jpeg', img)[1].tostring()
		
		num += 1
		if num > bestNum:
			num = 0
			recordFileNum += 1
			writer.close()
			recordFileName = ("train.tfrecords-%.3d" % recordFileNum)
			print("Creating: ", recordFileName)
			writer = tf.io.TFRecordWriter(os.path.join(recordPath, recordFileName))
		
		example = tf.train.Example(features=tf.train.Features(feature={
			"image/height": tf.train.Feature(int64_list=tf.train.Int64List(value=[size[1]])),
			"image/width": tf.train.Feature(int64_list=tf.train.Int64List(value=[size[0]])),
			"image/class/label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(class_name)])),
			"image/encoded": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
		}))
		
		writer.write(example.SerializeToString())
	writer.close()
	return


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input_dataset", type=str, required=True, help="Input dataset root directory")
	parser.add_argument("-o", "--output_dir", type=str, required=True, help="Output directory for tfrecords")
	parser.add_argument("-m", "--images_per_record", type=int, default=1000, help="Maximum number of image to store in one tf record" )
	args = parser.parse_args()
	main(args)