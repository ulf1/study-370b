import numpy as np
import pandas as pd
import gc
import sklearn.linear_model
import sklearn.ensemble
import os
import joblib


# load data
with open('./data/preprocessed.npy', 'rb') as fp:
    feats1 = np.load(fp)
    feats2 = np.load(fp)
    feats3 = np.load(fp)
    feats4 = np.load(fp)
    feats5 = np.load(fp)
    feats6 = np.load(fp)
    feats7 = np.load(fp)
    feats8 = np.load(fp)
    feats9 = np.load(fp)


df = pd.read_csv("./data/ratings.csv", encoding="ISO-8859-1")
y1mos = df["MOS_Complexity"].values
y2mos = df["MOS_Understandability"].values
y3mos = df["MOS_Lexical_difficulty"].values
del df
gc.collect()


# combine vectors
y_train = np.c_[y1mos, y2mos, y3mos]

x_train = np.hstack([
    feats1, feats2, feats3, feats4, feats5, feats6, feats7, feats9
])

del feats1, feats2, feats3, feats4, feats5, feats6, feats7, feats9
del y1mos, y2mos, y3mos
gc.collect()


# train linear model
model1a = sklearn.linear_model.BayesianRidge()
model1a.fit(X=feats8[:, 1].reshape(-1, 1), y=y_train[:, 0])

model1b = sklearn.linear_model.BayesianRidge()
model1b.fit(X=feats8[:, 1].reshape(-1, 1), y=y_train[:, 1])

model1c = sklearn.linear_model.BayesianRidge()
model1c.fit(X=feats8[:, 1].reshape(-1, 1), y=y_train[:, 2])

# compute residuals
eps1a = y_train[:, 0] - model1a.predict(feats8[:, 1].reshape(-1, 1))
eps1b = y_train[:, 1] - model1b.predict(feats8[:, 1].reshape(-1, 1))
eps1c = y_train[:, 2] - model1c.predict(feats8[:, 1].reshape(-1, 1))

# train random forest on residuals
model2 = sklearn.ensemble.RandomForestRegressor(
    n_estimators=100,
    max_depth=16,
    min_samples_leaf=10,
    bootstrap=True, oob_score=True, max_samples=0.5,
    random_state=42
)

model2.fit(X=x_train, y=np.c_[eps1a, eps1b, eps1c])

# save model
os.makedirs('./models/model7-rf/', exist_ok=True)
joblib.dump(model2, './models/model7-rf/model2.joblib')
joblib.dump(model1a, './models/model7-rf/model1a.joblib')
joblib.dump(model1b, './models/model7-rf/model1b.joblib')
joblib.dump(model1c, './models/model7-rf/model1c.joblib')
