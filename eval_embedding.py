import numpy as np
import matplotlib.pyplot as plt


from eval_method import oc_svm_classify, lof_detection, evaluate_embedding, isforest_classify, kde
from sklearn.manifold import TSNE


model_path = 'E:\Data\TraceCluster\log\\20epoch_CGConv_mean_0301data_0.6normal_randomview\\'

emb_train = np.load(model_path + 'emb_train.npy')
y_train = np.load(model_path + 'y_train.npy')
trace_class_train = np.load(model_path + 'trace_class_train.npy')
url_status_class_train = np.load(model_path + 'url_status_class_train.npy')
url_class_list_train = np.load(model_path + 'url_class_list_train.npy')

emb_test = np.load(model_path + 'emb_test.npy')
y_test = np.load(model_path + 'y_test.npy')
trace_class_test = np.load(model_path + 'trace_class_test.npy')
url_status_class_test = np.load(model_path + 'url_status_class_test.npy')
url_class_list_test = np.load(model_path + 'url_class_list_test.npy')

# oc_svm_classify(emb_train, emb_test, y_train, y_test, nu=0.01, kernel='poly')
# oc_svm_classify(emb_train, emb_test, y_train, y_test, nu=0.01, kernel='linear')
oc_svm_classify(emb_train, emb_test, y_train, y_test, nu=0.01, kernel='rbf')
# oc_svm_classify(emb_train, emb_test, y_train, y_test, nu=0.03, kernel='poly')
# oc_svm_classify(emb_train, emb_test, y_train, y_test, nu=0.03, kernel='linear')
oc_svm_classify(emb_train, emb_test, y_train, y_test, nu=0.03, kernel='rbf')
# oc_svm_classify(emb_train, emb_test, y_train, y_test, nu=0.05, kernel='poly')
# oc_svm_classify(emb_train, emb_test, y_train, y_test, nu=0.05, kernel='linear')
oc_svm_classify(emb_train, emb_test, y_train, y_test, nu=0.05, kernel='rbf')
# oc_svm_classify(emb_train, emb_test, y_train, y_test, nu=0.07, kernel='poly')
# oc_svm_classify(emb_train, emb_test, y_train, y_test, nu=0.07, kernel='linear')
oc_svm_classify(emb_train, emb_test, y_train, y_test, nu=0.07, kernel='rbf')
# oc_svm_classify(emb_train, emb_test, y_train, y_test, nu=0.09, kernel='poly')
# oc_svm_classify(emb_train, emb_test, y_train, y_test, nu=0.09, kernel='linear')
oc_svm_classify(emb_train, emb_test, y_train, y_test, nu=0.09, kernel='rbf')
# oc_svm_classify(emb_train, emb_test, y_train, y_test, nu=0.003, kernel='poly')
# oc_svm_classify(emb_train, emb_test, y_train, y_test, nu=0.003, kernel='linear')
# oc_svm_classify(emb_train, emb_test, y_train, y_test, nu=0.003, kernel='rbf')
# oc_svm_classify(emb_train, emb_test, y_train, y_test, nu=0.1, kernel='poly')
# oc_svm_classify(emb_train, emb_test, y_train, y_test, nu=0.1, kernel='linear')
# oc_svm_classify(emb_train, emb_test, y_train, y_test, nu=0.1, kernel='rbf')
# oc_svm_classify(emb_train, emb_test, y_train, y_test, nu=0.13, kernel='poly')
# oc_svm_classify(emb_train, emb_test, y_train, y_test, nu=0.13, kernel='linear')
oc_svm_classify(emb_train, emb_test, y_train, y_test, nu=0.13, kernel='rbf')
# oc_svm_classify(emb_train, emb_test, y_train, y_test, nu=0.15, kernel='poly')
# oc_svm_classify(emb_train, emb_test, y_train, y_test, nu=0.15, kernel='linear')
# oc_svm_classify(emb_train, emb_test, y_train, y_test, nu=0.15, kernel='rbf')

isforest_classify(emb_train, emb_test, y_train, y_test)
lof_detection(emb_train, emb_test, y_train, y_test, [])
# kde(emb_train, emb_test, y_train, y_test)
# evaluate_embedding(np.concatenate([emb_train, emb_test]), np.concatenate([y_train, y_test]), search=True)

# tsne = TSNE()
# data_embedding = np.concatenate([emb_train, emb_test])
# labels = np.concatenate([y_train, y_test])
# trace_class = np.concatenate([trace_class_train, trace_class_test])
# url_status_class = np.concatenate([url_status_class_train, url_status_class_test])
# url_class_list = np.concatenate([url_class_list_train, url_class_list_test])
# x = tsne.fit_transform(data_embedding)
# plt.scatter(x[:, 0], x[:, 1], c=labels, marker='o', s=10, cmap=plt.cm.Spectral)
# plt.show()
# plt.savefig(model_path + '/t-sne-test.jpg')