import pandas as pd
import numpy as np
import tensorflow as tf
import json
import gc
import argparse
from featureeng import preprocessing
from utils import get_random_mos
import keras_cor as kcor


parser = argparse.ArgumentParser()
parser.add_argument('--corr-trgt', type=int, default=1)
parser.add_argument('--corr-regul', type=int, default=1)
args = parser.parse_args()


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

# Correlation between outputs
y_rho = np.corrcoef(np.c_[y1mos, y2mos, y3mos], rowvar=False)

# Target correlations
target_corr = []
for i in range(y_rho.shape[0]):
    for j in range(1 + i, y_rho.shape[1]):
        target_corr.append(y_rho[i, j])
target_corr = tf.stack(target_corr)


# Feature Engineering
feats1, feats2, feats3, feats4, feats5, feats6 = preprocessing(texts)

print("Number of features")
print(f"{'Num SBert':>20s}: {feats1[0].shape[0]}")
print(f"{'Node vs Token dist.':>20s}: {feats2[0].shape[0]}")
print(f"{'PoS tag distr.':>20s}: {feats3[0].shape[0]}")
print(f"{'Morph. tags':>20s}: {feats4[0].shape[0]}")
print(f"{'Consonant cluster':>20s}: {feats5[0].shape[0]}")
print(f"{'Morph./lexemes':>20s}: {feats6[0].shape[0]}")


# Training set
def generator_trainingset(num_bootstrap: int = 64):
    num_data = len(feats1)
    for j in range(num_bootstrap):
        for i in range(num_data):
            # merge features
            xinputs = np.hstack([feats1[i], feats2[i], feats3[i], feats4[i], feats5[i], feats6[i]])

            # simulate noise targets
            y1, y2, y3 = get_random_mos(
                y1mos[i], y1std[i], y2mos[i], y2std[i], y3mos[i], y3std[i], 
                y_rho=y_rho, corr_trgt=args.corr_trgt)

            outputs = [y1, y2, y3]
            yield {
                "inputs": tf.cast(xinputs, dtype=tf.float32, name="inputs")
            }, {
                "outputs": tf.constant(outputs, dtype=tf.float32, name="outputs")
            }

batch_size = 128

dim_features = len(feats1[0]) + len(feats2[0]) + len(feats3[0]) + len(feats4[0]) + len(feats5[0]) + len(feats6[0])


ds_train = tf.data.Dataset.from_generator(
    lambda: generator_trainingset(
        num_bootstrap=64
    ),
    output_signature=(
        {
            "inputs": tf.TensorSpec(shape=(dim_features), dtype=tf.float32, name="inputs"),
        }, {
            "outputs": tf.TensorSpec(shape=(3), dtype=tf.float32, name="outputs")
        }
    )
).batch(batch_size)


# Validation set
def generator_validationset():
    num_data = len(feats1)
    for i in range(num_data):
        xinputs = np.hstack([feats1[i], feats2[i], feats3[i], feats4[i], feats5[i], feats6[i]])
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
).batch(batch_size)


# Modeling

# the input tensor
inputs = tf.keras.Input(shape=(dim_features,), name="inputs")

# Dense + 4.0
mos = tf.keras.layers.Dense(
    units=3, use_bias=False,
    kernel_initializer='glorot_uniform',
)(inputs)
mos = tf.keras.layers.Dense(
    units=3, use_bias=False,
    kernel_initializer='glorot_uniform',
)(mos)
mos = tf.keras.layers.Lambda(
    lambda s: s + tf.constant(4.0), name='mos'
)(mos)
if args.corr_regul == 1:
    mos = kcor.CorrOutputsRegularizer(target_corr, cor_rate=0.1)(mos)

# Function API model
model = tf.keras.Model(
    inputs=[inputs],
    outputs={"outputs": mos},
    name="scoring_model"
)


# Training
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        min_delta=1e-6,
        patience=10,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=f"best-model-370b-{args.corr_trgt}",
        monitor="val_loss",
        mode="min",
        save_best_only=True,
    ),
]

optimizer = tf.keras.optimizers.Adam(
    learning_rate=3e-4,  # Karpathy, 2019
    beta_1=.9, beta_2=.999, epsilon=1e-7,  # Kingma and Ba, 2014, p.2
    amsgrad=True  # Reddi et al, 2018, p.5-6
)

model.compile(
    optimizer=optimizer,
    loss={"outputs": "mean_squared_error"},
)

history = model.fit(
    ds_train, 
    validation_data=ds_valid,
    callbacks=callbacks,
    epochs=500,
)

with open(f"best-model-370b-{args.corr_trgt}/history.json", 'w') as fp:
    json.dump(history.history, fp)

