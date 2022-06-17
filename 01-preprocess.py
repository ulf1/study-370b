from featureeng import preprocessing
import pandas as pd
import numpy as np

# Load Dataset
df = pd.read_csv("data/ratings.csv", encoding="ISO-8859-1")
texts = df["Sentence"].values 
# Feature Engineering
feats1, feats2, feats3, feats4, feats5, feats6, feats7, feats8, feats9 = preprocessing(texts)
# save
with open('data/preprocessed.npy', 'wb') as fp:
    np.save(fp, feats1)
    np.save(fp, feats2)
    np.save(fp, feats3)
    np.save(fp, feats4)
    np.save(fp, feats5)
    np.save(fp, feats6)
    np.save(fp, feats7)
    np.save(fp, feats8)
    np.save(fp, feats9)


# inference
df = pd.read_csv("data2/validation_set.csv", encoding="utf-8")
texts = df["Sentence"].values 
feats1, feats2, feats3, feats4, feats5, feats6, feats7, feats8, feats9 = preprocessing(texts)
with open('data/validationset.npy', 'wb') as fp:
    np.save(fp, feats1)
    np.save(fp, feats2)
    np.save(fp, feats3)
    np.save(fp, feats4)
    np.save(fp, feats5)
    np.save(fp, feats6)
    np.save(fp, feats7)
    np.save(fp, feats8)
    np.save(fp, feats9)

