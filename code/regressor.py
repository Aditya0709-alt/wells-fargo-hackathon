from sklearn.base import BaseEstimator
from xgboost import sklearn
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

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
                            reg_alpha=1,
                            reg_lambda=0.5,
                            scale_pos_weight=1,
                            base_score=0.5,
                            seed=0,
                            missing=None)
        self.w = np.array([3.0521e-02,3.3850e-03,-2.7892e-02,9.4246e-02,1.2712e-01,5.6794e-02
,1.9702e-01,3.0102e-02,8.1020e-02,2.2443e-03,-3.6303e-02,-9.9930e-03
,-7.2356e-03,-6.6374e-03,7.2554e-02,-1.0639e-02,-8.9164e-02,-7.6698e-02
,-7.3221e-02,-2.6325e-02,1.5297e-02,-6.1099e-03,-1.6564e-02,1.1742e-03
,-7.7687e-03,-4.0734e-02,3.5347e-02,-8.9857e-03,-1.0205e-02,-3.5139e-02
,8.7736e-03,-2.6164e-02,-7.4057e-04,6.9800e-02,5.1630e-02,8.2260e-02
,-4.3334e-02,9.5439e-02,3.8949e-02,2.7576e-02,-2.7300e-02,-1.9236e-02
,1.3960e-02,-9.1715e-02,-8.0246e-02,1.6001e-01,-1.4912e-01,-1.1418e-01
,-1.3520e-01,5.8030e-02,1.8183e-01,-3.1726e-02,-7.4795e-02,-5.3430e-02
,-4.1667e-02,2.4433e-02,-1.5640e-02,-2.0981e-02,4.8331e-03,-2.2744e-02
,2.1778e-02,-9.1474e-03,-2.7065e-02,-1.3960e-03,3.1320e-02,2.4609e-02
,2.7434e-02,1.4061e-02,-3.9493e-03,1.7370e-02,5.4428e-03,4.9994e-03
,1.1100e-02,1.3571e-02,2.6117e-03,3.6254e-03,1.2581e-02,2.2057e-02
,-1.5871e-02,1.3411e-02,-1.6218e-02,-4.9300e-02,-4.8487e-02,-6.6901e-02
,-1.9708e-02,-3.6207e-02,2.7848e-02,3.3245e-02,-2.5913e-02,4.8864e-02
,1.7982e-02,7.2035e-02,9.8399e-03,-1.2854e-01,1.2498e-01,2.5496e-01
,4.8815e-01,1.2856e-02,2.7124e-02,-1.1177e-01,-6.9739e-02,-7.9357e-02
,-1.3767e-01,-3.4607e-02,-9.0663e-02,2.0239e-03,6.8687e-02,-2.8339e-02
,-2.3041e-02,7.7071e-03,-4.1781e-02,3.0516e-02,3.4045e-02,5.5087e-02
,5.4454e-02,1.6309e-02,1.5335e-03,1.3867e-02,1.8400e-02,3.6903e-03
,2.1292e-02,3.8298e-02,-3.4507e-02,2.0960e-03,3.4506e-03,1.3975e-02
,-2.4490e-02,2.9441e-02,-2.5951e-02,1.5139e-02,-4.7242e-02,-1.0273e-01
,8.0461e-03,-6.2661e-02,2.7466e-02,-4.3963e-03,-4.4565e-02,1.3144e-02
,-7.3661e-02,5.3355e-02,-3.5869e-03,-5.7825e-02,1.8184e-01,3.0521e-01
,4.2624e-01
])
        self.x = None
        self.y = None
        self.loss = lambda w : loss(w, self.x, self.y)
        self.grad = lambda w : grad(w, self.x, self.y)

    def fit(self, X, y):
        self.clf.fit(X, y)
        labels = y.reshape((-1,1))
        temp = X.copy()
        self.x = np.array(temp, order="F")
        self.y = np.array(labels, order="F")
        self.w = np.ones((X.shape[1], 1), order="F") / 100
        self.w, _, _ = fmin_l_bfgs_b(func=self.loss, x0=self.w, fprime=self.grad)

    def predict(self, X):
        predicted = self.clf.predict(X)
        predicted2 = X.dot(self.w)
        return np.maximum(0, ((predicted + predicted2)/2 + 0.5).astype(int)) # Average two predictions