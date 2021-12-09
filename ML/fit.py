import pickle
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

clf = MLPClassifier(max_iter=500, hidden_layer_sizes=(100,), solver='adam', alpha=0.0002)
with open('dataset/train.pkl', 'rb') as file:
    # 训练
    train_set = pickle.load(file)
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_set['X'])   # 标准化（MLP对此敏感）
    joblib.dump(scaler, 'model/scaler.model')
    train_label = train_set['y']
    clf.fit(train_data, train_label)
    joblib.dump(clf, 'model/mlp.model')  # 保存模型
