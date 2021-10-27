from src import Parameters
from src import Preprocessing
from src import TextClassifier
from src import Run
from src import RunEnsemble
import argparse
from pathlib import Path

class Controller(Parameters):
	
	def __init__(self, data_dir, num_per_file, size_tile):
		# Preprocessing pipeline
		self.data = self.prepare_data(data_dir, num_per_file, size_tile)

		# Initialize the model
		#self.model = TextClassifier(Parameters, size_tile)
		self.model = [TextClassifier(Parameters, size_tile) for i in range(2)]

		# Training - Evaluation pipeline
		#Run().train(self.model, self.data, Parameters)
		RunEnsemble().train(self.model, self.data, Parameters)

	@staticmethod
	def prepare_data(data_dir, num_per_file, size_tile):
		pr = Preprocessing(data_dir, num_per_file, size_tile)
		pr.load_data()
		pr.x_train, pr.y_train = pr.creating_train_dataset(pr.filenames_train, pr.y_train_full)
		pr.x_test, pr.y_test = pr.creating_train_dataset(pr.filenames_test, pr.y_test_full)
		pr.x_test_challenge = pr.padding_sentences(pr.filenames_test_challenge)
		pr.put_data_on_cuda()
		return {'x_train': pr.x_train, 'y_train': pr.y_train, 'x_test': pr.x_test, \
				'y_test': pr.y_test, 'x_test_challenge':pr.x_test_challenge, 'test_challenge_ids':pr.ids_test, \
				'data_dir':data_dir, 'num_per_file':pr.num_per_file, 'size_tile':pr.size_tile}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, type=Path,
                        help="directory where data is stored")
    parser.add_argument("--num_per_file", required=True, type=int,default=1,
                        help="directory where data is stored")
    parser.add_argument("--size_tile", required=True, type=int,default=1000,
                        help="directory where data is stored")
    
    args = parser.parse_args()
    assert args.data_dir.is_dir()
    controller = Controller(args.data_dir, args.num_per_file, args.size_tile)