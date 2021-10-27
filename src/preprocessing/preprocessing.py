import re
import os
import nltk
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize

class Preprocessing:
    
	def __init__(self, data_dir, num_per_file, size_tile):
		self.data_dir = data_dir
		self.num_per_file = num_per_file
		self.size_tile = size_tile
		self.x_padded = None
		self.x_raw = None
		self.y = None
		self.x_train = None
		self.x_test = None
		self.y_train = None
		self.y_test = None
		
	def load_data(self):
		# Reads the raw csv file and split into
		# sentences (x) and target (y)
		
		train_dir = self.data_dir / "train_input" / "resnet_features"
		test_dir = self.data_dir / "test_input"  / "resnet_features"

		train_output_filename = self.data_dir / "training_output.csv"
		train_output = pd.read_csv(train_output_filename)

        # Get the filenames for train
		#filenames_train = [train_dir / "{}.npy".format(idx) for idx in train_output["ID"]]
		self.filenames_train = [train_dir / "{}".format(elem) for elem in sorted(os.listdir(train_dir))] 
		for filename in self.filenames_train:
			assert filename.is_file(), filename

        # Get the labels
		labels_train = train_output["Target"].values
		assert len(self.filenames_train) == len(labels_train)

        # Get the numpy filenames for test
		self.filenames_test_challenge = sorted(test_dir.glob("*.npy"))
		for filename in self.filenames_test_challenge:
			assert filename.is_file(), filename
		self.ids_test = [f.stem for f in self.filenames_test_challenge]
		self.y = labels_train        
		self.filenames_train, self.filenames_test, self.y_train_full, self.y_test_full = \
				train_test_split(self.filenames_train, self.y, test_size=0.20, random_state=10)
		print(len(self.filenames_train), len(self.y_train_full))


	def creating_train_dataset(self, filenames, y):
		features, ys = [], []
		for k, f in enumerate(filenames):
			patient_features = np.load(f)
			if patient_features.shape[0]<1000:
				patient_features = np.transpose(np.tile(np.transpose(patient_features), reps=int(1000/patient_features.shape[0])+1)[:,:1000])
			patient_features = patient_features[:, 3:]
			for i in range(self.num_per_file):
				#indexes = np.sort(np.random.choice(1000, size=int(self.size_tile), replace=True))
				indexes = np.arange(i*int(1000/self.num_per_file), (i+1)*int(1000/self.num_per_file), 1)
				features.append(patient_features[indexes])
				ys.append(y[k])
		x, y = np.stack(features, axis=0), np.array(ys)
		return x, y

	def padding_sentences(self, filenames):
		features = []
		for f in filenames:
			patient_features = np.load(f)
			if patient_features.shape[0]<1000:
				patient_features = np.transpose(np.tile(np.transpose(patient_features), reps=int(1000/patient_features.shape[0])+1)[:,:1000])
			patient_features = patient_features[:, 3:]
			for i in range(self.num_per_file):
				#indexes = np.sort(np.random.choice(1000, size=int(self.size_tile), replace=True))
				indexes = np.arange(i*int(1000/self.num_per_file), (i+1)*int(1000/self.num_per_file), 1)
				features.append(patient_features[indexes])
		return np.stack(features, axis=0)

	def split_data(self, x, y, num_per_file):
		indexes = int(len(self.y)*0.80)*num_per_file
		self.x_train, self.x_test, self.y_train, self.y_test = x[:indexes], x[indexes:], y[:indexes], y[indexes:]
        
	def put_data_on_cuda(self):
		self.x_train = torch.from_numpy(self.x_train).cuda()
		self.x_test = torch.from_numpy(self.x_test).cuda()
		self.y_train = torch.from_numpy(self.y_train).cuda()
		self.y_test = torch.from_numpy(self.y_test).cuda()
		self.x_test_challenge = torch.from_numpy(self.x_test_challenge).cuda()