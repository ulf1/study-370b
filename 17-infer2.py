import numpy as np
import gc
import joblib
import pandas as pd
from zipfile import ZipFile
import os


# load inference dataset
with open('data/testset.npy', 'rb') as fp:
    feats1 = np.load(fp)
    feats2 = np.load(fp)
    feats3 = np.load(fp)
    feats4 = np.load(fp)
    feats5 = np.load(fp)
    feats6 = np.load(fp)
    feats7 = np.load(fp)
    feats8 = np.load(fp)
    feats9 = np.load(fp)

x_infer = np.hstack([
    feats1, feats2, feats3, feats4, feats5, feats6, feats7, feats9
])

del feats1, feats2, feats3, feats4, feats5, feats6, feats7, feats9
gc.collect()


# load model
model2 = joblib.load('./models/model7-rf/model2.joblib')
model1a = joblib.load('./models/model7-rf/model1a.joblib')
model1b = joblib.load('./models/model7-rf/model1b.joblib')
model1c = joblib.load('./models/model7-rf/model1c.joblib')


# inference
eps = model2.predict(x_infer)
y1_infer = eps[:, 0] + model1a.predict(feats8[:, 1].reshape(-1, 1))
y2_infer = eps[:, 1] + model1b.predict(feats8[:, 1].reshape(-1, 1))
y3_infer = eps[:, 2] + model1c.predict(feats8[:, 1].reshape(-1, 1))

# save 
df = pd.read_csv("./data2/part2_public.csv", encoding="utf-8")
df["MOS"] = np.maximum(1.0, np.minimum(7.0, y1_infer))
df[["ID", "MOS"]].to_csv("./models/model7-rf/answer.csv", index=False)

# zip upload file
with ZipFile("./models/model7-rf/answer7-test.csv.zip", 'w') as zipf:
    zipf.write("./models/model7-rf/answer.csv", arcname="answer.csv")

# remove answer.csv
os.remove("./models/model7-rf/answer.csv")


# linear model
y1_pred = model1a.predict(feats8[:, 1].reshape(-1, 1))
# save 
df = pd.read_csv("./data2/part2_public.csv", encoding="utf-8")
df["MOS"] = np.maximum(1.0, np.minimum(7.0, y1_pred))
df[["ID", "MOS"]].to_csv("./models/model7-rf/answer.csv", index=False)
# zip upload file
with ZipFile("./models/model7-rf/answer7a-test.csv.zip", 'w') as zipf:
    zipf.write("./models/model7-rf/answer.csv", arcname="answer.csv")
# remove answer.csv
os.remove("./models/model7-rf/answer.csv")

