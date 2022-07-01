import numpy as np
import gc
import pandas as pd
from zipfile import ZipFile
import os
import tensorflow as tf


# load inference dataset
with open('data/validationset.npy', 'rb') as fp:
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
    feats1, feats2, feats3, feats4, feats5, feats6, feats7, feats8, feats9
])

del feats1, feats2, feats3, feats4, feats5, feats6, feats7, feats8, feats9
gc.collect()


# load model
model = tf.keras.models.load_model("./models/model4-sp")


# inference
y_infer = model.predict(x_infer)


# save 
df = pd.read_csv("./data2/validation_set.csv", encoding="utf-8")
df["MOS"] = np.maximum(1.0, np.minimum(7.0, y_infer["outputs"][:, 0]))
df[["ID", "MOS"]].to_csv("./models/model4-sp/answer.csv", index=False)


# zip upload file
with ZipFile("./models/model4-sp/answer4-sp.csv.zip", 'w') as zipf:
    zipf.write("./models/model4-sp/answer.csv", arcname="answer.csv")
