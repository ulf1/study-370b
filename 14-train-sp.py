import numpy as np
import pandas as pd
import gc
import os
import maxjoshua as mh
import tensorflow as tf
import sklearn.preprocessing
import json
from utils import get_random_mos
import sparsity_pattern
import random


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
y1std = df["Std_Complexity"].values
y2mos = df["MOS_Understandability"].values
y2std = df["Std_Understandability"].values
y3mos = df["MOS_Lexical_difficulty"].values
y3std = df["Std_Lexical_difficulty"].values
del df
gc.collect()

# dims
dim_features = len(feats1[0]) + len(feats2[0]) + len(feats3[0]) \
    + len(feats4[0]) + len(feats5[0]) + len(feats6[0]) \
    + len(feats7[0]) + len(feats8[0]) + len(feats9[0])


# Correlation between outputs
rho = np.corrcoef(np.c_[y1mos, y2mos, y3mos], rowvar=False)


# sample proba
def lookup_proba(x, bnd, proba):
    for i in range(len(bnd) - 1):
        if x >= bnd[i] and x <= bnd[i + 1]:
            return proba[i]
    return 0.0

pdf1, bnd1 = np.histogram(y1mos, density=True, bins=5)
psmp1 = 1. / pdf1
psmp1 = psmp1 / psmp1.sum()
p1 = [lookup_proba(val, bnd1, psmp1) for val in y1mos]

pdf2, bnd2 = np.histogram(y2mos, density=True, bins=5)
psmp2 = 1. / pdf2
psmp2 = psmp2 / psmp2.sum()
p2 = [lookup_proba(val, bnd2, psmp2) for val in y2mos]

pdf3, bnd3 = np.histogram(y3mos, density=True, bins=5)
psmp3 = 1. / pdf3
psmp3 = psmp3 / psmp3.sum()
p3 = [lookup_proba(val, bnd3, psmp3) for val in y3mos]

psmp = np.c_[p1, p2, p3].sum(axis=1)
psmp = psmp / psmp.sum()


# Training set
def generator_trainingset():
    # simulate noise targets
    y1, y2, y3 = get_random_mos(
        y1mos, y1std, y2mos, y2std, y3mos, y3std, 
        adjust=0.1, rho=rho, corr_trgt=1)
    # subsample 384 items
    idx = np.random.choice(
        range(len(feats1)), p=psmp, replace=False, size=512)

    # loop over subsample
    for i in idx:
        # merge features
        xinputs = np.hstack([
            feats1[i], feats2[i], feats3[i], 
            feats4[i], feats5[i], feats6[i],
            feats7[i], feats8[i], feats9[i]
        ])
        outputs = [y1[i], y2[i], y3[i]]
        # outputs = [y1mos[i], y2mos[i], y3mos[i]]
        yield {
            "inputs": tf.cast(xinputs, dtype=tf.float32, name="inputs")
        }, {
            "outputs": tf.constant(outputs, dtype=tf.float32, name="outputs")
        }


ds_train = tf.data.Dataset.from_generator(
    lambda: generator_trainingset(),
    output_signature=(
        {
            "inputs": tf.TensorSpec(shape=(dim_features), dtype=tf.float32, name="inputs"),
        }, {
            "outputs": tf.TensorSpec(shape=(3), dtype=tf.float32, name="outputs")
        }
    )
).batch(128)


# Validation set
def generator_validationset():
    num_data = len(feats1)
    for i in range(num_data):
        xinputs = np.hstack([
            feats1[i], feats2[i], feats3[i], 
            feats4[i], feats5[i], feats6[i],
            feats7[i], feats8[i], feats9[i]
        ])
        outputs = [y1mos[i], y2mos[i], y3mos[i]]
        yield {
            "inputs": tf.cast(xinputs, dtype=tf.float32, name="inputs"),
        }, {
            "outputs": tf.constant(outputs, dtype=tf.float32, name="outputs")
        }

ds_valid = tf.data.Dataset.from_generator(
    lambda: generator_validationset(),
    output_signature=(
        {
            "inputs": tf.TensorSpec(shape=(dim_features), dtype=tf.float32, name="inputs"),
        }, {
            "outputs": tf.TensorSpec(shape=(3), dtype=tf.float32, name="outputs")
        }
    )
).batch(128)



# random sparsity patterns
random.seed(42)
np.random.seed(42)

# layer 1
indices1 = sparsity_pattern.get(
    'random', r=dim_features, c=256, pct=0.012)  # 3x per row/col
values1 = np.random.normal(size=(len(indices1),))
values1 = (values1 / np.abs(values1).sum()).tolist()

# layer 2
indices2 = sparsity_pattern.get(
    'random', r=256, c=256, pct=0.012)  # 3x per row/col
values2 = np.random.normal(size=(len(indices2),))
values2 = (values2 / np.abs(values2).sum()).tolist()

# layer 3
indices3 = sparsity_pattern.get(
    'random', r=256, c=256, pct=0.012)  # 3x per row/col
values3 = np.random.normal(size=(len(indices3),))
values3 = (values3 / np.abs(values3).sum()).tolist()


# (A) inputs
inputs = tf.keras.Input(shape=(dim_features,), name="inputs")

# (B) layer 1
h = mh.SparseLayerAsEnsemble(
    num_in=dim_features, 
    num_out=256, 
    sp_indices=indices1, 
    sp_values=values1,
    sp_trainable=True,
    norm_trainable=False
)(inputs)
h = tf.keras.layers.Activation("swish")(h)

# layer 2
h2 = mh.SparseLayerAsEnsemble(
    num_in=256, 
    num_out=256, 
    sp_indices=indices2, 
    sp_values=values2,
    sp_trainable=True,
    norm_trainable=False
)(h)
h2 = tf.keras.layers.Activation("swish")(h2)
h2 = tf.keras.layers.Dropout(0.5)(h2)
h = tf.keras.layers.Add()([h, h2])

# layer 3
h2 = mh.SparseLayerAsEnsemble(
    num_in=256, 
    num_out=256, 
    sp_indices=indices2, 
    sp_values=values2,
    sp_trainable=True,
    norm_trainable=False
)(h)
h2 = tf.keras.layers.Activation("swish")(h2)
h2 = tf.keras.layers.Dropout(0.5)(h2)
h = tf.keras.layers.Add()([h, h2])

# (C) final layer
h = tf.keras.layers.Dense(
    units=3, use_bias=True,
    kernel_initializer='glorot_uniform'
)(h)

# (D) Function API model
model = tf.keras.Model(
    inputs=[inputs],
    outputs={"outputs": h},
    name="scoring_model"
)
# compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=3e-4, beta_1=.9, beta_2=.999, epsilon=1e-7, amsgrad=True),
    loss={"outputs": "mean_squared_error"},
)

# train meta model
os.makedirs('./models', exist_ok=True)
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        min_delta=1e-6,
        patience=100,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath="./models/model4-sp",
        monitor="loss",
        mode="min",
        save_best_only=True,
    ),
]

history = model.fit(
    ds_train, 
    validation_data=ds_valid,
    callbacks=callbacks,
    epochs=5000,
)

with open("./models/model4-sp/history.json", 'w') as fp:
    json.dump(history.history, fp)

# save model
# os.makedirs('./models', exist_ok=True)
# model.save('./models/model4-sp')
