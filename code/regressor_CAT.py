from sklearn.base import BaseEstimator
from xgboost import sklearn
import numpy as np
from sklearn.linear_model import Lasso
from scipy.optimize import fmin_l_bfgs_b

n_group = 7

def loss(w, x, y):
    y_pred = x.dot(w).reshape((-1, 1))
    y_pred = np.array(y_pred, order="F")
    return np.mean(np.exp(0.1 * (y - y_pred)) - 0.1 * (y - y_pred) - 1)

def grad(w, x, y):
    y_pred = x.dot(w).reshape((-1, 1))
    y_pred = np.array(y_pred, order="F")
    gradient = 0.1 * (1 - np.exp(0.1 * (y - y_pred))) * x
    return np.mean(gradient, axis=0)

def obj(y_true, y_pred):
    grad = 0.1 * (1 - np.exp(0.1 * (y_true - y_pred)))
    hess = 0.01 * np.exp(0.1*(y_true - y_pred))
    return grad, hess



class Regressor(BaseEstimator):
    def __init__(self):
        self.clf = sklearn.XGBRegressor(max_depth=3,
                            learning_rate=0.1,
                            n_estimators=300,
                            silent=True,
                            objective=obj,
                            gamma=0,
                            min_child_weight=1,
                            max_delta_step=0,
                            subsample=1,
                            colsample_bytree=1,
                            colsample_bylevel=0.25,
                            reg_alpha=0 ,
                            reg_lambda=0.5,
                            scale_pos_weight=1,
                            base_score=0.5,
                            seed=0,
                            missing=None)
        self.w=np.array([np.array([1.7029e-02,1.3079e-01,6.1581e-02,-1.6783e-02,3.3474e-02
,-2.2277e-02,-2.1690e-01,1.1374e-01,7.1316e-02,3.6111e-02,-1.9211e-01
,8.9843e-02,1.0525e-02,-8.8967e-02,-1.6134e-01,-1.0343e-01,3.8159e-02
,1.2840e-02,1.4358e-01,-1.2254e-01,1.4967e-01,3.8851e-02,8.4922e-02
,2.1995e-02,-1.7713e-01,4.5296e-02,5.0263e-02,3.5791e-05,-1.4180e-01
,1.5155e-01,-7.8438e-02,-1.0855e-01,-1.0028e-01,-5.2810e-02,7.0936e-02
,8.6607e-02,6.8758e-02,-1.7710e-01,3.1382e-02,2.7970e-01,3.8615e-01
,2.0975e-01,1.1192e-02,-3.1998e-01,1.9952e-01,4.5477e-01,-6.7926e-02
,-1.2770e-01,8.1820e-02,1.7651e-01,3.3767e-02,3.8274e-01,8.7390e-03
,-4.5134e-02,-5.6199e-02,-8.8637e-02,7.9332e-02,-1.0147e-01,1.7228e-01
,-6.2791e-02,2.2888e-03,5.2206e-02,1.0851e-01,3.7676e-02,1.0128e-01
,1.0922e-02,-1.9359e-01,6.2475e-02,-5.5140e-02,2.9518e-02,-2.3585e-02
,-1.1021e-01,1.2358e-01,3.9869e-03,-3.0878e-02,-2.9022e-02,-2.5127e-02
,-5.1951e-02,6.4713e-02,6.3186e-02,4.3845e-02,-3.2788e-02,8.0593e-03
,6.9834e-02,-5.3207e-02,8.0649e-02,-7.0133e-02,-1.1874e-01,-2.0268e-01
,3.6341e-02,-2.8456e-02,2.5505e-01,-5.9185e-02,-1.6351e-01,2.0862e-01
,3.9112e-01,-1.7588e-02,3.9111e-02,2.9766e-01,5.3394e-01,-4.8566e-03
,6.3414e-02,2.7350e-01,-1.6731e-01,-2.6914e-02,-1.9693e-01,1.4585e-01
,4.4899e-02,-3.2440e-02,4.4213e-02,1.1280e-01,2.1263e-01,1.1246e-01
,-5.3757e-02,-1.4070e-01,8.6012e-02,-1.2140e-01,7.1008e-04,1.3947e-02
,-2.5169e-02,1.7305e-01,-3.6080e-02,-6.7890e-02,9.9060e-02,4.4189e-02
,-1.1350e-01,1.4912e-01,3.4591e-02,5.1782e-02,1.5098e-02,8.5624e-03
,-1.0366e-01,-6.0745e-02,1.7117e-01,-5.4439e-02,-1.2122e-01,-2.8721e-01
,-2.1258e-01,3.5069e-02,8.1284e-02,-2.1620e-01,-3.0161e-01
]) for i in range(n_group)])
        self.x = None
        self.y = None
        self.loss = lambda w : loss(w, self.x, self.y)
        self.grad = lambda w : grad(w, self.x, self.y)
        self.select_loss = lambda i: (lambda w: loss(w, self.x[i], self.y[i]))
        self.select_grad = lambda i: (lambda w: grad(w, self.x[i], self.y[i]))
        

        

    def fit(self, X, y):
        self.clf.fit(X, y)
        labels = y.reshape((-1,1))
        temp = X.copy()
        self.x = np.array([np.array([temp[i, 1:] for i in range(temp.shape[0]) if temp[i, 0]==j+1], order="F") 
                            for j in range(n_group)], order="F")
        self.y = np.array([np.array([labels[i] for i in range(temp.shape[0]) if temp[i, 0]==j+1], order="F") 
                            for j in range(n_group)], order="F")
        for i in range(n_group):
            print("Training Group " + str(i))
            self.w[i, :], _, _ = fmin_l_bfgs_b(func=self.select_loss(i), x0=self.w[i, :], 
                                                fprime=self.select_grad(i))#, iprint=100)


    def predict(self, X):
        predicted = self.clf.predict(X)
        predicted2 = np.zeros(X.shape[0])
        for i in range(n_group):
            predicted2[X[:, 0] == i+1] = X[X[:, 0] == i+1][:, 1:].dot(self.w[i, :])
        return np.maximum(0, ((predicted + predicted2)/2 + 0.5).astype(int))