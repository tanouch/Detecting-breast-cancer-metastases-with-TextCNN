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

class Run:
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
		
		# Define optimizer
		optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
        
		# Starts training phase
		model.train()
		model.cuda()
        
		for epoch in range(params.epochs):
			predictions = []
			losses = []

			for x_batch, y_batch in loader_train:
				y_batch = y_batch.type(torch.FloatTensor).cuda()
				
				# Feed the model
				x_batch = torch.transpose(x_batch, 1, 2)
				y_pred = model(x_batch, "training")
				loss = F.binary_cross_entropy(y_pred, y_batch)
                
				# Clean gradientes
				optimizer.zero_grad()

				# Gradients calculation
				loss.backward()
				losses.append(loss.detach().cpu().numpy())
				
				# Gradients update
				optimizer.step()
				
				# Save predictions
				predictions += list(y_pred.detach().cpu().numpy())
			
			# Metrics calculation
			if epoch%20==0:
				train_pred_mean, train_pred_max, train_pred_min = get_statistics(predictions, npf)
				test_pred, test_pred_mean, test_pred_max, test_pred_min = Run.evaluation(model, loader_test, npf)
				X = np.stack((train_pred_mean, train_pred_max, train_pred_min))
				Xtest = np.stack((test_pred_mean, test_pred_max, test_pred_min))
				clf, train_log_pred, test_log_pred = Run.train_logistic_regression(X, ytrain_per_slide, Xtest)
                
				train_acc = Run.calculate_accuray(data['y_train'], predictions)
				test_acc = Run.calculate_accuray(data['y_test'], test_pred)
				train_auc = roc_auc_score(data['y_train'].detach().cpu().numpy(), np.array(predictions))
				test_auc = roc_auc_score(data['y_test'].detach().cpu().numpy(), np.array(test_pred))
				train_auc_mean = roc_auc_score(ytrain_per_slide, train_pred_mean)
				test_auc_mean = roc_auc_score(ytest_per_slide, test_pred_mean)
				train_auc_logreg = roc_auc_score(ytrain_per_slide, train_log_pred)
				test_auc_logreg = roc_auc_score(ytest_per_slide, test_log_pred)

				print("")                
				print("Epoch: %d, loss: %.5f, Train accuracy: %.5f, Test accuracy: %.5f" % (epoch, np.mean(np.array(losses)), train_acc, test_acc))
				print("Train auc: %.5f, Test auc: %.5f" % (train_auc, test_auc))
				print("Train auc mean: %.5f, Test auc mean: %.5f" % (train_auc_mean, test_auc_mean))
				print("Train auc logreg: %.5f, Test auc logreg: %.5f" % (train_auc_logreg, test_auc_logreg))
                
		test_pred, _, _ = Run.evaluation_test_challenge(model, loader_test_challenge, npf)
		#Xtest = np.transpose(np.stack((test_pred_mean, test_pred_max, test_pred_min)))
		#test_pred = clf.predict_proba(Xtest)[:,1]
		assert np.max(test_pred) <= 1.0
		assert np.min(test_pred) >= 0.0
		ids_number_test = [i.split("ID_")[1] for i in data['test_challenge_ids']]
		test_output = pd.DataFrame({"ID": ids_number_test, "Target": test_pred})
		test_output.set_index("ID", inplace=True)
		test_output.to_csv(data['data_dir'] / "preds_test_baseline.csv")

	@staticmethod
	def train_logistic_regression(X, y, Xtest):
		X, Xtest = np.transpose(logit(X)), np.transpose(logit(Xtest))
		clf = LogisticRegression(penalty='none').fit(X, y)
		train_predictions = clf.predict_proba(X)[:,1]
		test_predictions = clf.predict_proba(Xtest)[:,1]
		return clf, train_predictions, test_predictions
        
	def evaluation(model, loader_test, npf):
		model.eval()
		predictions = []
		with torch.no_grad():
			for x_batch, y_batch in loader_test:
				x_batch = torch.transpose(x_batch, 1, 2)
				y_pred = model(x_batch, "test")
				predictions += list(y_pred.detach().cpu().numpy())
		model.train()
		predictions_mean, predictions_max, predictions_min = get_statistics(predictions, npf)
		return predictions, predictions_mean, predictions_max, predictions_min

	def evaluation_test_challenge(model, loader_test, npf):
		model.eval()
		predictions = []
		with torch.no_grad():
			for x_batch in loader_test:
				x_batch = torch.transpose(x_batch, 1, 2)
				y_pred = model(x_batch, "test")
				predictions += list(y_pred.detach().cpu().numpy())
		predictions_mean, predictions_max, predictions_min = get_statistics(predictions, npf)
		return predictions_mean, predictions_max, predictions_min
    
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
	return predictions_mean, predictions_max, predictions_min
