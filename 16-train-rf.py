import numpy as np
import pandas as pd
import gc
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

del feats1, feats2, feats3, feats4, feats5, feats6, feats7, feats8, feats9
del y1mos, y2mos, y3mos
gc.collect()


# training
model = sklearn.ensemble.RandomForestRegressor(
    n_estimators=100,
    max_depth=16,
    min_samples_leaf=10,
    bootstrap=True, oob_score=True, max_samples=0.5,
    random_state=42
)

model.fit(X=x_train, y=y_train)

# save model
os.makedirs('./models/model6-rf/', exist_ok=True)
joblib.dump(model, './models/model6-rf/model.joblib')




