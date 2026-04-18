import numpy as np
import scipy.io as io 
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn import metrics
import sys
import pickle


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def load_sysdata(data_name):
	W = list()
	X = list()

	if 'ImgGendata' in data_name:
		data = load_obj(data_name)
		X.append(np.array(data['X1']))
		X.append(np.array(data['X2']))
		X.append(np.array(data['Y']))

		

		return X, W

	data = io.loadmat('./generatedData/'+data_name)

	if 'simData3_' in data_name or 'simCFData3' in data_name or 'simCFHData3' in data_name:
		X.append(data['X1'])
		X.append(data['X2'])
		X.append(data['X3'])
		X.append(data['Y'])
		W.append(data['u1'])
		W.append(data['u2'])
		W.append(data['u3'])
		W.append(data['v0'])
	elif 'example_data' in data_name or 'sim' in data_name:
		X.append(data['X'])
		X.append(data['Y'])
		W.append(data['u0'])
		W.append(data['v0'])
	elif data_name == 'synDataAda':
		n_views = data['Data_X'].shape[1]
		for i_view in range(n_views):
			X.append(data['Data_X'][0,i_view])
			W.append(data['GroundTruth_W'][0,i_view])

	elif 'ImgGendata' in data_name:
		X.append(data['X1'])
		X.append(data['X2'])
		X.append(data['Y'])
	return X, W


def evaluate_AUC(trueW, predW):
	n_views = len(trueW)

	anslist = list()

	for i_view in range(n_views):
		fpr, tpr, thresholds = metrics.roc_curve(trueW[i_view],predW[i_view])
		anslist.append(metrics.auc(fpr,tpr))
	return anslist

