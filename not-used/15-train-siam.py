import pandas as pd
import numpy as np
import tensorflow as tf
import json
import gc
import argparse

from utils import get_random_mos
import maxjoshua as mh
import sparsity_pattern
import random
import os

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=128)
args = parser.parse_args()


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


# Masks for 6x6 buckets
def get_buckets(x1, x2, x3):
    b1 = np.logical_and(x1 >= x2, x2 >= x3)
    b2 = np.logical_and(x1 >= x3, x3 >= x2)
    b3 = np.logical_and(x2 >= x1, x1 >= x3)
    b4 = np.logical_and(x2 >= x3, x3 >= x1)
    b5 = np.logical_and(x3 >= x1, x1 >= x2)
    b6 = np.logical_and(x3 >= x2, x2 >= x1)
    return b1, b2, b3, b4, b5, b6

buckets_mu = get_buckets(y1mos, y2mos, y3mos)
buckets_sd = get_buckets(y1std, y2std, y3std)

bucket_indicies = []
for idxm in range(6):
    for idxs in range(6):
        bucket_indicies.append(
            np.where(np.logical_and(buckets_mu[idxm], buckets_sd[idxs]))[0])



# Dataset Generator for Siamese Net
def draw_example_index(bucket_indicies):
    # draw example IDs i,j from buckets
    bi, bj = np.random.choice(range(36), size=2)
    idxi = bucket_indicies[bi]
    idxj = bucket_indicies[bj]
    # pick one of the three targets
    ok = np.random.choice(range(3), size=1)[0]
    by = y1mos if ok == 0 else y2mos if ok == 1 else y3mos
    byi, byj = by[idxi], by[idxj]
    # swap if needed
    cuti, cutj = byi.mean(), byj.mean()
    if cuti < cutj:
        cuti, cutj = cutj, cuti
        byi, byj = byj, byi
        idxi, idxj = idxj, idxi
    # draw positive example
    wi = byi >= cuti
    wi = wi / wi.sum()
    if np.isnan(wi).any():
        wi = byi / byi.sum()
    i = np.random.choice(idxi, p=wi, size=1)[0]
    # draw negative example
    wj = byj < cutj
    wj = wj / wj.sum()
    if np.isnan(wj).any():
        wj = (7.0 - byj) / byj.sum()
    j = np.random.choice(idxj, p=wj, size=1)[0]
    # done
    return i, j, ok


def generator_trainingset(num_draws: int = 512):
    # simulate noise targets
    y1pos, y2pos, y3pos = get_random_mos(
        y1mos, y1std, y2mos, y2std, y3mos, y3std, 
        adjust=0.1, rho=rho, corr_trgt=1)

    y1neg, y2neg, y3neg = get_random_mos(
        y1mos, y1std, y2mos, y2std, y3mos, y3std, 
        adjust=0.1, rho=rho, corr_trgt=1)

    for _ in range(num_draws):
        # draw example IDs i,j from buckets
        i, j, ok = draw_example_index(bucket_indicies)

        # merge features
        x0pos = np.hstack([
            feats1[i], feats2[i], feats3[i], 
            feats4[i], feats5[i], feats6[i],
            feats7[i], feats8[i], feats9[i]
        ])
        x0neg = np.hstack([
            feats1[j], feats2[j], feats3[j], 
            feats4[j], feats5[j], feats6[j],
            feats7[j], feats8[j], feats9[j]
        ])
        x0aug = x0pos * (1. + np.random.normal(
            size=x0pos.shape, loc=0, scale=1e-6))

        # compute differences
        d1 = np.abs(y1mos[i] - y1mos[j])
        d2 = np.abs(y2mos[i] - y2mos[j])
        d3 = np.abs(y3mos[i] - y3mos[j])
        # d1 = np.abs(y1pos[i] - y1neg[j])
        # d2 = np.abs(y2pos[i] - y2neg[j])
        # d3 = np.abs(y3pos[i] - y2neg[j])

        # concat targets
        targets = [
            y1pos[i], y2pos[i], y3pos[i], 
            y1neg[j], y2neg[j], y3neg[j], 
            d1, d2, d3, ok]
        yield {
            "inp_pos": tf.cast(x0pos, dtype=tf.float32, name="inp_pos"),
            "inp_neg": tf.cast(x0neg, dtype=tf.float32, name="inp_neg"),
            "inp_aug": tf.cast(x0aug, dtype=tf.float32, name="inp_aug"),
        }, tf.constant(targets, dtype=tf.float32, name="targets")


ds_train = tf.data.Dataset.from_generator(
    lambda: generator_trainingset(
        num_draws=512  # 512 keep it small to regenerate noise more often
    ),
    output_signature=(
        {
            "inp_pos": tf.TensorSpec(shape=(dim_features), dtype=tf.float32, name="inp_pos"),
            "inp_neg": tf.TensorSpec(shape=(dim_features), dtype=tf.float32, name="inp_neg"),
            "inp_aug": tf.TensorSpec(shape=(dim_features), dtype=tf.float32, name="inp_aug"),
        },
        tf.TensorSpec(shape=(10), dtype=tf.float32, name="targets"),
    )
).batch(args.batch_size).prefetch(1)


# Validation set
def generator_validationset(num_draws=16384):
    for _ in range(num_draws):
        # draw example IDs i,j from buckets
        i, j, ok = draw_example_index(bucket_indicies)

        # merge features
        x0pos = np.hstack([
            feats1[i], feats2[i], feats3[i], 
            feats4[i], feats5[i], feats6[i],
            feats7[i], feats8[i], feats9[i]
        ])
        x0neg = np.hstack([
            feats1[j], feats2[j], feats3[j], 
            feats4[j], feats5[j], feats6[j],
            feats7[j], feats8[j], feats9[j]
        ])
        x0aug = x0pos * (1. + np.random.normal(
            size=x0pos.shape, loc=0, scale=1e-6))

        # compute differences
        d1 = y1mos[i] - y1mos[j]
        d2 = y2mos[i] - y2mos[j]
        d3 = y3mos[i] - y3mos[j]

        # concat targets
        targets = [
            y1mos[i], y2mos[i], y3mos[i], 
            y1mos[j], y2mos[j], y3mos[j], 
            d1, d2, d3, ok]
        yield {
            "inp_pos": tf.cast(x0pos, dtype=tf.float32, name="inp_pos"),
            "inp_neg": tf.cast(x0neg, dtype=tf.float32, name="inp_neg"),
            "inp_aug": tf.cast(x0aug, dtype=tf.float32, name="inp_aug"),
        }, tf.constant(targets, dtype=tf.float32, name="targets")


ds_valid = tf.data.Dataset.from_generator(
    lambda: generator_validationset(num_draws=16384),
    output_signature=(
        {
            "inp_pos": tf.TensorSpec(shape=(dim_features), dtype=tf.float32, name="inp_pos"),
            "inp_neg": tf.TensorSpec(shape=(dim_features), dtype=tf.float32, name="inp_neg"),
            "inp_aug": tf.TensorSpec(shape=(dim_features), dtype=tf.float32, name="inp_aug"),
        },
        tf.TensorSpec(shape=(10), dtype=tf.float32, name="targets"),
    )
).batch(16384)

xy_valid = list(ds_valid)[0]  # generate once!


# Build the Scoring Model
def build_scoring_model(dim_features: int,
                        sp_pct: float = 0.0061,
                        fc_units: int = 3,
                        fc_bias_mu: float = 4.0):
    # random sparsity patterns
    random.seed(42)
    np.random.seed(42)

    # layer 1
    indices1 = sparsity_pattern.get(
        'random', r=dim_features, c=dim_features, pct=sp_pct)  # 3x per row/col
    values1 = np.random.normal(size=(len(indices1),))
    values1 = (values1 / np.abs(values1).sum()).tolist()

    # layer 2
    indices2 = sparsity_pattern.get(
        'random', r=dim_features, c=dim_features, pct=sp_pct)  # 3x per row/col
    values2 = np.random.normal(size=(len(indices2),))
    values2 = (values2 / np.abs(values2).sum()).tolist()

    # layer 3
    indices3 = sparsity_pattern.get(
        'random', r=dim_features, c=dim_features, pct=sp_pct)  # 3x per row/col
    values3 = np.random.normal(size=(len(indices3),))
    values3 = (values3 / np.abs(values3).sum()).tolist()

    # (A) inputs
    inputs = tf.keras.Input(shape=(dim_features,), name="inputs")

    # (B) layer 1
    h1 = mh.SparseLayerAsEnsemble(
        num_in=dim_features, 
        num_out=dim_features, 
        sp_indices=indices1, 
        sp_values=values1,
        sp_trainable=True,
        norm_trainable=False
    )(inputs)
    h1 = tf.keras.layers.Activation("swish")(h1)
    h1 = tf.keras.layers.Add()([h1, inputs])

    # layer 2
    h2 = mh.SparseLayerAsEnsemble(
        num_in=dim_features, 
        num_out=dim_features, 
        sp_indices=indices2, 
        sp_values=values2,
        sp_trainable=True,
        norm_trainable=False
    )(h1)
    h2 = tf.keras.layers.Activation("swish")(h2)
    h2 = tf.keras.layers.Add()([h2, h1])

    # layer 3
    h3 = mh.SparseLayerAsEnsemble(
        num_in=dim_features, 
        num_out=dim_features, 
        sp_indices=indices3, 
        sp_values=values3,
        sp_trainable=True,
        norm_trainable=False
    )(h2)
    h3 = tf.keras.layers.Activation("swish")(h3)
    h3 = tf.keras.layers.Add()([h3, h2])
    # h3 = tf.keras.layers.Add()([inputs, h3])

    # (C) final layer
    out = tf.keras.layers.Dense(
        units=fc_units, use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer=tf.keras.initializers.Constant(value=fc_bias_mu)
    )(h3)

    # (D) Function API model
    model = tf.keras.Model(
        inputs=[inputs],
        outputs=[out, h3],
        name="scoring_model"
    )
    # done
    return model


# Build the Siamese Net
def cosine_distance(a, b, tol=1e-8):
    """ Cosine distance """
    n = tf.reduce_sum(tf.multiply(a, b), axis=1)
    da = tf.math.sqrt(tf.reduce_sum(tf.math.pow(a, 2), axis=1))
    db = tf.math.sqrt(tf.reduce_sum(tf.math.pow(b, 2), axis=1))
    return 1.0 - (n / (da * db + tol))


def cosine_distance_normalized(a, b, tol=1e-8):
    """ Cosine distance with normalized input vectors """
    a = tf.math.l2_normalize(a, axis=1)
    b = tf.math.l2_normalize(b, axis=1)
    return cosine_distance(a, b, tol=tol)


def loss1_rank_triplet(y_true, y_pred):
    """ Triplet ranking loss between last representation layers """
    # the diffs must be normed to [0,1] by dividing by 6
    ok = tf.cast(y_true[:, -1], dtype=tf.int8)
    margin = tf.math.multiply(
        tf.math.abs(y_true[:, 6]) / 6.0, 
        tf.cast(ok == 0, dtype=tf.float32))
    margin += tf.math.multiply(
        tf.math.abs(y_true[:, 7]) / 6.0,
        tf.cast(ok == 1, dtype=tf.float32))
    margin += tf.math.multiply(
        tf.math.abs(y_true[:, 8]) / 6.0,
        tf.cast(ok == 2, dtype=tf.float32))
    # read model outputs
    n = (y_pred.shape[1] - 9) // 3
    repr_pos = y_pred[:, 9:(9 + n)]
    repr_neg = y_pred[:, (9 + n):(9 + n * 2)]
    repr_aug = y_pred[:, (9 + n * 2):]
    # Triplet ranking loss between last representation layers
    dist1 = cosine_distance_normalized(repr_pos, repr_aug)
    dist2 = cosine_distance_normalized(repr_pos, repr_neg)
    loss = tf.reduce_mean(tf.math.maximum(0.0, dist1 - dist2 + margin))
    return loss


def loss2_mse_target(y_true, y_pred):
    """ MSE loss on positive or negative noise targets """
    # norm to [0,1] by dividing by 6
    ok = tf.cast(y_true[:, -1], dtype=tf.int8)
    loss  = tf.reduce_mean(tf.math.pow(tf.math.multiply(
        (y_true[:, 0] - y_pred[:, 0]) / 6.0, 
        tf.cast((ok == 1) | (ok == 2), dtype=tf.float32)), 2))
    loss += tf.reduce_mean(tf.math.pow(tf.math.multiply(
        (y_true[:, 3] - y_pred[:, 3]) / 6.0, 
        tf.cast((ok == 1) | (ok == 2), dtype=tf.float32)), 2))
    loss += tf.reduce_mean(tf.math.pow(tf.math.multiply(
        (y_true[:, 1] - y_pred[:, 1]) / 6.0, 
        tf.cast((ok == 0) | (ok == 2), dtype=tf.float32)), 2))
    loss += tf.reduce_mean(tf.math.pow(tf.math.multiply(
        (y_true[:, 4] - y_pred[:, 4]) / 6.0, 
        tf.cast((ok == 0) | (ok == 2), dtype=tf.float32)), 2))
    loss += tf.reduce_mean(tf.math.pow(tf.math.multiply(
        (y_true[:, 2] - y_pred[:, 2]) / 6.0, 
        tf.cast((ok == 0) | (ok == 1), dtype=tf.float32)), 2))
    loss += tf.reduce_mean(tf.math.pow(tf.math.multiply(
        (y_true[:, 5] - y_pred[:, 5]) / 6.0, 
        tf.cast((ok == 0) | (ok == 1), dtype=tf.float32)), 2))
    return loss


def loss_total(y_true, y_pred):
    loss = .9 * loss1_rank_triplet(y_true, y_pred)
    loss += .1 * loss2_mse_target(y_true, y_pred)
    return loss


def build_siamese_net(dim_features: int):
    # the input tensors
    inp_pos = tf.keras.Input(shape=(dim_features,), name='inp_pos')
    inp_neg = tf.keras.Input(shape=(dim_features,), name='inp_neg')
    inp_aug = tf.keras.Input(shape=(dim_features,), name='inp_aug')

    # linstantiate the shared scoring model
    scorer = build_scoring_model(
        dim_features=dim_features
    )

    # predict examples for each input
    y_pos, repr_pos = scorer(inp_pos)
    y_neg, repr_neg = scorer(inp_neg)
    _, repr_aug = scorer(inp_aug)

    # Function API model
    model = tf.keras.Model(
        inputs={
            'inp_pos': inp_pos,
            'inp_neg': inp_neg,
            'inp_aug': inp_aug,
        },
        outputs=tf.keras.backend.concatenate([
            y_pos, y_neg,
            repr_pos, repr_neg, repr_aug
        ], axis=1),
        name="scorer_contrast"
    )

    # optimization settings
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=3e-4,  # Karpathy, 2019
            beta_1=.9, beta_2=.999, epsilon=1e-7,  # Kingma and Ba, 2014, p.2
            amsgrad=True  # Reddi et al, 2018, p.5-6
        ),
        loss=[loss_total],
        metrics=[loss_total, loss1_rank_triplet, loss2_mse_target],
    )

    return model


# Training
os.makedirs('./models', exist_ok=True)
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        min_delta=1e-6,
        patience=50,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath="./models/model5-siam",
        monitor="val_loss",
        mode="min",
        save_best_only=True,
    ),
]


model = build_siamese_net(dim_features)


history = model.fit(
    ds_train, 
    validation_data=xy_valid,
    callbacks=callbacks,
    epochs=1000,
    verbose=1
)


with open("./models/model5-siam/history.json", 'w') as fp:
    json.dump(history.history, fp)

