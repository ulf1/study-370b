from featureeng import preprocessing
import pandas as pd
import tensorflow as tf
import numpy as np

# Load Dataset
df = pd.read_csv("data2/validation_set.csv", encoding="utf-8")
texts = df["Sentence"].values 

# Feature Engineering
feats1, feats2, feats3, feats4, feats5, feats6 = preprocessing(texts)
xinputs = np.hstack([feats1, feats2, feats3, feats4, feats5, feats6])


# Load model
model = tf.keras.models.load_model(
    "best-model-370b-1-1", compile=False)
# Predict
y_pred = model.predict(xinputs)
# read only the the 1st output neuron
df["MOS"] = np.maximum(1.0, np.minimum(7.0, y_pred['outputs'][:, 0]))
# save
df[["ID", "MOS"]].to_csv("data2/answer111.csv", index=False)


# Load model
model_siamese = tf.keras.models.load_model(
    "best-model-370b-siamese-1-1", compile=False)
model = model_siamese.layers[2]
# Predict
y_pred = model.predict(xinputs)
# read only the the 1st output neuron
df["MOS"] = y_pred[0][:, 0]
# save
df[["ID", "MOS"]].to_csv("data2/answer211.csv", index=False)


# Load model
model = tf.keras.models.load_model(
    "best-model-370c-1-1", compile=False)
# Predict
y_pred = model.predict(xinputs)
# read only the the 1st output neuron
df["MOS"] = y_pred['mos'][:, 0]
# save
df[["ID", "MOS"]].to_csv("data2/answer311.csv", index=False)


# Load model
model = tf.keras.models.load_model(
    "best-model-370c-1-0", compile=False)
# Predict
y_pred = model.predict(xinputs)
# read only the the 1st output neuron
df["MOS"] = y_pred['mos'][:, 0]
# save
df[["ID", "MOS"]].to_csv("data2/answer310.csv", index=False)


# Load model
model = tf.keras.models.load_model(
    "best-model-370c-0-1", compile=False)
# Predict
y_pred = model.predict(xinputs)
# read only the the 1st output neuron
df["MOS"] = y_pred['mos'][:, 0]
# save
df[["ID", "MOS"]].to_csv("data2/answer301.csv", index=False)

