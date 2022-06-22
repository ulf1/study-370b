import numpy as np
import pandas as pd
import gc
import os
import maxjoshua as mh
import tensorflow as tf
import sklearn.preprocessing
import json


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
    feats1, feats2, feats3, feats4, feats5, feats6, feats7, feats8, feats9
])

del feats1, feats2, feats3, feats4, feats5, feats6, feats7, feats8, feats9
del y1mos, y2mos, y3mos
gc.collect()


# pretrain submodels
# - always scale the inputs and targets -
indices, values, num_in, num_out = mh.pretrain_submodels(
    sklearn.preprocessing.scale(x_train), 
    sklearn.preprocessing.scale(y_train), 
    num_out=256, n_select=3,
)


# specify meta model
model = tf.keras.models.Sequential([
    # sub-models
    mh.SparseLayerAsEnsemble(
        num_in=num_in, 
        num_out=num_out, 
        sp_indices=indices, 
        sp_values=values,
        sp_trainable=False
    ),
    # meta model
    tf.keras.layers.Dense(
        units=3, use_bias=False,
        kernel_constraint=tf.keras.constraints.NonNeg()
    ),
    # scale up
    mh.InverseTransformer(
        units=3,
        init_bias=y_train.mean(), 
        init_scale=y_train.std()
    )
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=3e-4, beta_1=.9, beta_2=.999, epsilon=1e-7, amsgrad=True),
    loss='mean_squared_error'
)

# train meta model
os.makedirs('./models', exist_ok=True)
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath="./models/model2-sp",
        monitor="loss",
        mode="min",
        save_best_only=True,
    ),
]

history = model.fit(
    x_train, y_train, 
    epochs=500,
    callbacks=callbacks
)

with open("./models/model2-sp/history.json", 'w') as fp:
    json.dump(history.history, fp)

# save model
# os.makedirs('./models', exist_ok=True)
# model.save('./models/model2-sp')
