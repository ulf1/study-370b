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
parser.add_argument('--batch-size', type=int, default=128)
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
target_corr = tf.cast(target_corr, dtype=tf.float32)

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
            yield {
                "inputs": tf.cast(xinputs, dtype=tf.float32, name="inputs")
            }, {
                "clf1": tf.constant(np.int32(y1mos[i]) - 1, dtype=tf.int32, name="clf1"),
                "clf2": tf.constant(np.int32(y2mos[i]) - 1, dtype=tf.int32, name="clf2"),
                "clf3": tf.constant(np.int32(y3mos[i]) - 1, dtype=tf.int32, name="clf3"),
                "mos": tf.constant([y1, y2, y3], dtype=tf.float32, name="mos")
            }


dim_features = len(feats1[0]) + len(feats2[0]) + len(feats3[0]) + len(feats4[0]) + len(feats5[0]) + len(feats6[0])


ds_train = tf.data.Dataset.from_generator(
    lambda: generator_trainingset(
        num_bootstrap=64
    ),
    output_signature=(
        {
            "inputs": tf.TensorSpec(shape=(dim_features), dtype=tf.float32, name="inputs"),
        }, {
            "clf1": tf.TensorSpec(shape=(), dtype=tf.int32, name="clf1"),
            "clf2": tf.TensorSpec(shape=(), dtype=tf.int32, name="clf2"),
            "clf3": tf.TensorSpec(shape=(), dtype=tf.int32, name="clf3"),
            "mos": tf.TensorSpec(shape=(3), dtype=tf.float32, name="mos")
        }
    )
).batch(args.batch_size)



# Validation set
def generator_validationset():
    num_data = len(feats1)
    for i in range(num_data):
        xinputs = np.hstack([feats1[i], feats2[i], feats3[i], feats4[i], feats5[i], feats6[i]])
        outputs = [y1mos[i], y2mos[i], y3mos[i]]
        yield {
            "inputs": tf.cast(xinputs, dtype=tf.float32, name="inputs"),
        }, {
            "clf1": tf.constant(np.int32(y1mos[i]) - 1, dtype=tf.int32, name="clf1"),
            "clf2": tf.constant(np.int32(y2mos[i]) - 1, dtype=tf.int32, name="clf2"),
            "clf3": tf.constant(np.int32(y3mos[i]) - 1, dtype=tf.int32, name="clf3"),
            "mos": tf.constant(outputs, dtype=tf.float32, name="mos")
        }

ds_valid = tf.data.Dataset.from_generator(
    lambda: generator_validationset(),
    output_signature=(
        {
            "inputs": tf.TensorSpec(shape=(dim_features), dtype=tf.float32, name="inputs"),
        }, {
            "clf1": tf.TensorSpec(shape=(), dtype=tf.int32, name="clf1"),
            "clf2": tf.TensorSpec(shape=(), dtype=tf.int32, name="clf2"),
            "clf3": tf.TensorSpec(shape=(), dtype=tf.int32, name="clf3"),
            "mos": tf.TensorSpec(shape=(3), dtype=tf.float32, name="mos")
        }
    )
).batch(args.batch_size)


# Modeling
def build_scorer_classifer_layer(input_dims: int):
    # the input tensor
    inputs = tf.keras.Input(shape=(input_dims,), name="scorer_classifier_input")
    # (a) final layer for classification and regression
    h = tf.keras.layers.Dense(
        units=6, use_bias=False,
        kernel_initializer='glorot_uniform',
    )(inputs)
    # (a) Softmax for classification
    out_clf = tf.keras.layers.Activation("softmax")(h)
    # (b) Regression layer
    h2 = tf.keras.layers.Dense(
        units=1, use_bias=False,
        kernel_initializer='glorot_uniform',
    )(h)
    h2 = tf.keras.layers.Activation("tanh")(h2)
    h2 = tf.keras.layers.Lambda(lambda s: s * 0.5)(h2)
    # (b) Add classification label to regression layer
    h3 = tf.keras.layers.Lambda(
        lambda s: tf.reduce_sum(tf.multiply(
            s, tf.constant([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])), axis=-1)
    )(out_clf)
    out_rgr = tf.keras.layers.Add()([h2, h3])
    # return model
    return tf.keras.Model(
        inputs=[inputs],
        outputs=[out_clf, out_rgr])


# Function API model
inputs = tf.keras.Input(shape=(dim_features,), name="inputs")
clf1, rgr1 = build_scorer_classifer_layer(dim_features)(inputs)
clf2, rgr2 = build_scorer_classifer_layer(dim_features)(inputs)
clf3, rgr3 = build_scorer_classifer_layer(dim_features)(inputs)

mos = tf.concat([rgr1, rgr2, rgr3], axis=-1)
if args.corr_regul == 1:
    mos = kcor.CorrOutputsRegularizer(target_corr, cor_rate=0.1)(mos)

model = tf.keras.Model(
    inputs=[inputs],
    outputs={
        "clf1": clf1, "clf2": clf2, "clf3": clf3,
        "mos": mos
    },
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
        filepath=f"best-model-370c-{args.corr_trgt}-{args.corr_regul}",
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

class WeightedCategoricalLoss(tf.keras.losses.SparseCategoricalCrossentropy):
    def __init__(self, costs):
        super(WeightedCategoricalLoss, self).__init__()
        self.costs = costs
    def __call__(self, y_true, y_pred, sample_weight=None):
        weights = tf.gather_nd(params=self.costs, indices=y_true)
        weights = weights / tf.math.reduce_sum(weights)
        return super(WeightedCategoricalLoss, self).__call__(
            y_true, y_pred, sample_weight=weights)

_, cnt1 = np.unique(np.int32(y1mos) - 1, return_counts=True)
cost1 = tf.constant(cnt1.sum() / cnt1)

_, cnt2 = np.unique(np.int32(y2mos) - 1, return_counts=True)
cost2 = tf.constant(cnt2.sum() / cnt2)

_, cnt3 = np.unique(np.int32(y3mos) - 1, return_counts=True)
cost3 = tf.constant(cnt3.sum() / cnt3)

model.compile(
    optimizer=optimizer,
    loss={
        "clf1": WeightedCategoricalLoss(cost1),
        "clf2": WeightedCategoricalLoss(cost2),
        "clf3": WeightedCategoricalLoss(cost3),
        "mos": "mean_squared_error"
    },
)

history = model.fit(
    ds_train, 
    validation_data=ds_valid,
    callbacks=callbacks,
    epochs=5,
)

with open(f"best-model-370c-{args.corr_trgt}-{args.corr_regul}/history.json", 'w') as fp:
    json.dump(history.history, fp)

