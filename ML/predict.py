import joblib
import pickle
from sklearn.neural_network import MLPClassifier


clf = joblib.load('model/mlp.model')
scaler = joblib.load('model/scaler.model')
with open('dataset/test.pkl', 'rb') as infile:
    test_set = pickle.load(infile)
    pred = clf.predict(scaler.transform(test_set['X']))
    with open('mlp_pred.txt', 'w') as outfile:
        for x in pred:
            outfile.write(str(x) + '\n')
