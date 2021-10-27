import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from scipy.special import logit

class DatasetMaper_testchallenge(Dataset):
	def __init__(self, x):
		self.x = x
		
	def __len__(self):
		return len(self.x)
		
	def __getitem__(self, idx):
		return self.x[idx]

class DatasetMaper(Dataset):
	def __init__(self, x, y):
		self.x = x
		self.y = y
		
	def __len__(self):
		return len(self.x)
		
	def __getitem__(self, idx):
		return self.x[idx], self.y[idx]

class RunEnsemble:
	'''Training, evaluation and metrics calculation'''

	@staticmethod
	def train(model, data, params):
		# Initialize dataset maper
		train = DatasetMaper(data['x_train'], data['y_train'])
		test = DatasetMaper(data['x_test'], data['y_test'])
		test_challenge = DatasetMaper_testchallenge(data['x_test_challenge'])
		ytrain, ytest, npf = data['y_train'].detach().cpu().numpy(), data['y_test'].detach().cpu().numpy(), data['num_per_file']
		ytrain_per_slide = np.array([ytrain[int(npf*i)] for i in range(int(len(ytrain)/npf))])
		ytest_per_slide = np.array([ytest[int(npf*i)] for i in range(int(len(ytest)/npf))])

		# Initialize loaders
		loader_train = DataLoader(train, batch_size=params.batch_size)
		loader_test = DataLoader(test, batch_size=params.batch_size)
		loader_test_challenge = DataLoader(test_challenge, batch_size=params.batch_size)
		
		optimizers = list()
		for i, m in enumerate(model):
			m.train()
			m.cuda()
			optimizers.append(optim.Adam(m.parameters(), lr=params.learning_rate))
        
		for epoch in range(params.epochs):
			predictions = []
			losses = []

			for x_batch, y_batch in loader_train:
				y_batch = y_batch.type(torch.FloatTensor).cuda()
				x_batch = torch.transpose(x_batch, 1, 2)
                
				for i, m in enumerate(model):
					indexes = torch.from_numpy(np.random.choice(len(y_batch), size=int(0.90*len(y_batch)), replace=False)).cuda()
					y_pred = m(x_batch[indexes])
					loss = F.binary_cross_entropy(y_pred, y_batch[indexes])
					optimizers[i].zero_grad()
					loss.backward()
					losses.append(loss.detach().cpu().numpy())
					optimizers[i].step()

			# Metrics calculation
			if epoch%30==0:
				train_pred, train_pred_mean, train_pred_max, train_pred_min = RunEnsemble.evaluation_train(model, loader_train, npf)
				test_pred, test_pred_mean, test_pred_max, test_pred_min = RunEnsemble.evaluation(model, loader_test, npf)
				train_acc = RunEnsemble.calculate_accuray(data['y_train'], train_pred)
				test_acc = RunEnsemble.calculate_accuray(data['y_test'], test_pred)
				train_auc = roc_auc_score(data['y_train'].detach().cpu().numpy(), train_pred)
				test_auc = roc_auc_score(data['y_test'].detach().cpu().numpy(), test_pred)
				train_auc_mean = roc_auc_score(ytrain_per_slide, train_pred_mean)
				test_auc_mean = roc_auc_score(ytest_per_slide, test_pred_mean)

				print("")                
				print("Epoch: %d, loss: %.5f, Train accuracy: %.5f, Test accuracy: %.5f" % (epoch, np.mean(np.array(losses)), train_acc, test_acc))
				print("Train auc: %.5f, Test auc: %.5f" % (train_auc, test_auc))
				print("Train auc mean: %.5f, Test auc mean: %.5f" % (train_auc_mean, test_auc_mean))
                
				_, test_pred, _, _ = RunEnsemble.evaluation_test_challenge(model, loader_test_challenge, npf)
				assert np.max(test_pred) <= 1.0
				assert np.min(test_pred) >= 0.0
				ids_number_test = [i.split("ID_")[1] for i in data['test_challenge_ids']]
				test_output = pd.DataFrame({"ID": ids_number_test, "Target": test_pred})
				test_output.set_index("ID", inplace=True)
				namefile = "preds_test_baseline_" + str(epoch)+"dropout95.csv"
				test_output.to_csv(data['data_dir'] / namefile)

	@staticmethod
	def evaluation_train(model, loader_test, npf):
		predictions = []
		with torch.no_grad():
			for x_batch, y_batch in loader_test:
				x_batch = torch.transpose(x_batch, 1, 2)
				y_pred = torch.cat([torch.unsqueeze(m(x_batch), 0) for m in model], dim=0)
				y_pred = torch.mean(y_pred, dim=0)
				predictions += list(y_pred.detach().cpu().numpy())
		return get_statistics(predictions, npf)

	def evaluation(model, loader_test, npf):
		for m in model:
			m.eval()
		predictions = []
		with torch.no_grad():
			for x_batch, y_batch in loader_test:
				x_batch = torch.transpose(x_batch, 1, 2)
				y_pred = torch.cat([torch.unsqueeze(m(x_batch), 0) for m in model], dim=0)
				y_pred = torch.mean(y_pred, dim=0)
				predictions += list(y_pred.detach().cpu().numpy())
		for m in model:
			m.train()
		return get_statistics(predictions, npf)

	def evaluation_test_challenge(model, loader_test, npf):
		for m in model:
			m.eval()
		predictions = []
		with torch.no_grad():
			for x_batch in loader_test:
				x_batch = torch.transpose(x_batch, 1, 2)
				y_pred = torch.cat([torch.unsqueeze(m(x_batch), 0) for m in model], dim=0)
				y_pred = torch.mean(y_pred, dim=0)
				predictions += list(y_pred.detach().cpu().numpy())
		return get_statistics(predictions, npf)
    
	@staticmethod
	def calculate_accuray(grand_truth, predictions):
		# Metrics calculation
		true_positives = 0
		true_negatives = 0
		for true, pred in zip(grand_truth, predictions):
			if (pred >= 0.5) and (true == 1):
				true_positives += 1
			elif (pred < 0.5) and (true == 0):
				true_negatives += 1
			else:
				pass
		# Return accuracy
		return (true_positives+true_negatives) / len(grand_truth)
    
def get_statistics(predictions, npf):
	predictions = np.stack(predictions, axis=0)
	predictions_mean = np.array([np.mean([predictions[npf*i:npf*i+npf]]) for i in range(int(len(predictions)/npf))])
	predictions_max = np.array([np.amax([predictions[npf*i:npf*i+npf]]) for i in range(int(len(predictions)/npf))])
	predictions_min = np.array([np.amin([predictions[npf*i:npf*i+npf]]) for i in range(int(len(predictions)/npf))])
	return predictions, predictions_mean, predictions_max, predictions_min
