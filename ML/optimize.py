import pickle
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

clf = MLPClassifier()
mlp_clf_tuned_parameters = {"hidden_layer_sizes": [(100,), (100, 30), (100, 30, 30)],
                            "solver": ['adam', 'sgd', 'lbfgs'],
                            "max_iter": [20],   # 为了快速得出最优参数这里把迭代次数设置小一点
                            "alpha": np.linspace(0.0001, 0.0005, 5)
                            }   # mlp参数调整范围
opt = GridSearchCV(clf, mlp_clf_tuned_parameters, n_jobs=6)     # 自动调参器
with open('dataset/train.pkl', 'rb') as file:
    train_set = pickle.load(file)
    scaler = StandardScaler()
    # 标准化（MLP对此敏感）
    train_data = scaler.fit_transform(train_set['X'])
    train_label = train_set['y']
    opt.fit(train_data, train_label)
    print(opt.get_params().keys())
    print(opt.best_params_)
