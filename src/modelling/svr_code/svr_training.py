import numpy as np
from sklearn.metrics import log_loss
from cuml.svm import SVC

base_path = "/home/mithil/PycharmProjects/lmsys-scoring"
train_file = np.load("/home/mithil/PycharmProjects/lmsys-scoring/data/hidden_states_train.npz")
train_hidden_states = train_file['hidden_states_all']
train_label = train_file['labels']
valid_file = np.load("/home/mithil/PycharmProjects/lmsys-scoring/data/hidden_states_eval.npz")
valid_hidden_states = valid_file['hidden_states_all']
valid_label = valid_file['labels']

svc = SVC(probability=True,C=25,tol=1e-4)
svc.fit(train_hidden_states, train_label,)
preds = svc.predict_proba(valid_hidden_states)
log_loss(y_true=valid_label,y_pred=preds)