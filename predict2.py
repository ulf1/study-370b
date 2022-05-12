import tensorflow as tf
from featureeng import preprocessing
import pandas as pd
import gc
import numpy as np


# Load the raw data
df = pd.read_csv("data/ratings.csv", encoding="ISO-8859-1")
# Input- & Output-Variables
texts = df["Sentence"].values 
y1mos = df["MOS_Complexity"].values
y1std = df["Std_Complexity"].values
y2mos = df["MOS_Understandability"].values
y2std = df["Std_Understandability"].values
y3mos = df["MOS_Lexical_difficulty"].values
y3std = df["Std_Lexical_difficulty"].values
# free memory
del df
gc.collect()

# Feature engineering
feats1, feats2, feats3, feats4, feats5, feats6 = preprocessing(texts)
# delete_models()

# Concat
xinputs = np.hstack([feats1, feats2, feats3, feats4, feats5, feats6])


def loss1_rank_triplet(a,b):
    return 0.0

def loss2_mse_diffs(a,b):
    return 0.0

def loss3_mse_target(a,b):
    return 0.0

model = tf.keras.models.load_model(
    "best-model-370b-siamese-1", 
    custom_objects={
        "loss1_rank_triplet": loss1_rank_triplet, 
        "loss2_mse_diffs": loss2_mse_diffs,
        "loss3_mse_target": loss3_mse_target
})

model_scoring = model.layers[2]
model_scoring.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=3e-4,  # Karpathy, 2019
        beta_1=.9, beta_2=.999, epsilon=1e-7,  # Kingma and Ba, 2014, p.2
        amsgrad=True  # Reddi et al, 2018, p.5-6
    ),
    loss='mean_absolute_error',
    metrics=['mean_absolute_error']
)

# Inference
y1hat, y2hat, y3hat, reprl = model_scoring.predict(xinputs)

# Compute MAEs
loss1 = tf.math.reduce_mean(tf.math.abs(y1hat - y1mos))
loss2 = tf.math.reduce_mean(tf.math.abs(y2hat - y2mos))
loss3 = tf.math.reduce_mean(tf.math.abs(y3hat - y3mos))

print("MAE:", loss1, loss2, loss3)

y_rho = np.corrcoef(np.c_[y1hat.numpy(), y2hat.numpy(), y3hat.numpy()], rowvar=False)
print(y_rho)
