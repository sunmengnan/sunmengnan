import os
import gzip
import shutil
import struct
import numpy as np
from six.moves import urllib

DATA_SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def downloadMNIST(filename):
	"""
	download dataset from Yann's website
	:param filename: dataset name
	:param path: dataset working directory
	:return: none
	"""
	work_path = os.path.join(BASE_DIR,'datasets')
	if not os.path.exists(work_path):
		os.mkdir(work_path)
	filepath = os.path.join(work_path,filename)
	if not os.path.exists(filepath):
		filepath, _ = urllib.request.urlretrieve(DATA_SOURCE_URL+filename,filepath)
		stat_info 	= os.stat(filepath)
		print('Dataset downloaded, ',filename,stat_info.st_size, 'bytes. ',end='')
	filename = filename.replace('.gz','')
	if not os.path.exists(os.path.join(work_path,filename)):
		with gzip.open(filepath,'rb') as f_in:
			with open(os.path.join(work_path,filename),'wb') as f_out:
				shutil.copyfileobj(f_in,f_out)
				print('Dataset unzipped. ')

def loadMNIST(dataset="training", num_image=1):
	"""
	# loads data from MNIST datasets
	:param dataset:train or test
	:param num_image: ubyte file, either image or label
	:return:numpy matrix
	"""
	if dataset == "training":
		image_file = os.path.join(BASE_DIR, "datasets/train-images-idx3-ubyte")
		label_file = os.path.join(BASE_DIR, "datasets/train-labels-idx1-ubyte")
	if dataset == "test":
		image_file = os.path.join(BASE_DIR, "datasets/t10k-images-idx3-ubyte")
		label_file = os.path.join(BASE_DIR, "datasets/t10k-labels-idx1-ubyte")
	with open(image_file, 'br') as fd:  # b for binary, r for read only
		struct.unpack('>BBBB', fd.read(4))
		dataset_image_num, num_col, num_row = struct.unpack('>III', fd.read(12))
		image_buf = fd.read(num_image * num_col * num_row)
		image_data = np.frombuffer(image_buf, dtype=np.uint8).astype(np.float32)
		image_data = image_data.reshape(num_image, num_col, num_row)
	with open(label_file, 'br') as fd:
		magic = fd.read(4)
		struct.unpack('>BBBB', magic)
		struct.unpack('>I', fd.read(4))
		label_buf = fd.read(num_image)
		label_data = np.frombuffer(label_buf, dtype=np.uint8).astype(np.int32)
	return image_data, label_data

