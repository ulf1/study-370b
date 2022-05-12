import pandas as pd
import numpy as np
import tensorflow as tf
import json
import gc
import argparse
from featureeng import preprocessing, delete_models


parser = argparse.ArgumentParser()
parser.add_argument('--corr-trgt', type=int, default=0)
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

# Correlation between outputs
y_rho = np.corrcoef(np.c_[y1mos, y2mos, y3mos], rowvar=False)

# free memory
del df
gc.collect()


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


# Feature Engineering
# preprocess all examples
feats1, feats2, feats3, feats4, feats5, feats6 = preprocessing(texts)
delete_models()

print("Number of features")
print(f"{'Num SBert':>20s}: {feats1[0].shape[0]}")
print(f"{'Node vs Token dist.':>20s}: {feats2[0].shape[0]}")
print(f"{'PoS tag distr.':>20s}: {feats3[0].shape[0]}")
print(f"{'Morph. tags':>20s}: {feats4[0].shape[0]}")
print(f"{'Consonant cluster':>20s}: {feats5[0].shape[0]}")
print(f"{'Morph./lexemes':>20s}: {feats6[0].shape[0]}")



# Dataset Generator for Siamese Net
def get_random_mos(y1m, y1s, y2m, y2s, y3m, y3s, corr_trgt=0):
    if corr_trgt == 0:
        # simulate uncorrelated random scores
        y1 = np.random.normal(loc=y1m, scale=y1s, size=1)
        y2 = np.random.normal(loc=y2m, scale=y2s, size=1)
        y3 = np.random.normal(loc=y3m, scale=y3s, size=1)
        y1 = np.maximum(1.0, np.minimum(7.0, y1))[0]
        y2 = np.maximum(1.0, np.minimum(7.0, y2))[0]
        y3 = np.maximum(1.0, np.minimum(7.0, y3))[0]
    elif corr_trgt == 1:
        # simulate correlated random scores
        Y = np.random.standard_normal((1, 3))
        Y = np.dot(Y, np.linalg.cholesky(y_rho).T)[0]
        y1 = np.maximum(1.0, np.minimum(7.0, Y[0] * y1s + y1m))
        y2 = np.maximum(1.0, np.minimum(7.0, Y[1] * y2s + y2m))
        y3 = np.maximum(1.0, np.minimum(7.0, Y[2] * y3s + y3m))
    return y1, y2, y3


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
    return i, j


def generator_trainingset(num_draws: int = 65536):
    for _ in range(num_draws):
        # draw example IDs i,j from buckets
        i, j = draw_example_index(bucket_indicies)

        # merge features
        x0pos = np.hstack(
            [feats1[i], feats2[i], feats3[i], feats4[i], feats5[i], feats6[i]])
        x0neg = np.hstack(
            [feats1[j], feats2[j], feats3[j], feats4[j], feats5[j], feats6[j]])
            
        # simulate noise targets
        y1pos, y2pos, y3pos = get_random_mos(
            y1mos[i], y1std[i], y2mos[i], y2std[i], y3mos[i], y3std[i], 
            corr_trgt=args.corr_trgt)
        y1neg, y2neg, y3neg = get_random_mos(
            y1mos[j], y1std[j], y2mos[j], y2std[j], y3mos[j], y3std[j], 
            corr_trgt=args.corr_trgt)

        # compute differences
        d1 = y1mos[i] - y1mos[j]
        d2 = y2mos[i] - y2mos[j]
        d3 = y3mos[i] - y3mos[j]

        # concat targets
        targets = [y1pos, y2pos, y3pos, y1neg, y2neg, y3neg, d1, d2, d3]
        yield {
            "inp_pos": tf.cast(x0pos, dtype=tf.float32, name="inp_pos"),
            "inp_neg": tf.cast(x0neg, dtype=tf.float32, name="inp_neg"),
        }, tf.constant(targets, dtype=tf.float32, name="targets")


batch_size = 128

dim_features = len(feats1[0]) + len(feats2[0]) + len(feats3[0]) + len(feats4[0]) + len(feats5[0]) + len(feats6[0])

ds_train = tf.data.Dataset.from_generator(
    lambda: generator_trainingset(
        num_draws=65536
    ),
    output_signature=(
        {
            "inp_pos": tf.TensorSpec(shape=(dim_features), dtype=tf.float32, name="inp_pos"),
            "inp_neg": tf.TensorSpec(shape=(dim_features), dtype=tf.float32, name="inp_neg"),
        },
        tf.TensorSpec(shape=(9), dtype=tf.float32, name="targets"),
    )
).batch(batch_size).prefetch(1)


# Validation set
def generator_validationset(num_draws=65536):
    for _ in range(num_draws):
        # draw example IDs i,j from buckets
        i, j = draw_example_index(bucket_indicies)

        # merge features
        x0pos = np.hstack(
            [feats1[i], feats2[i], feats3[i], feats4[i], feats5[i], feats6[i]])
        x0neg = np.hstack(
            [feats1[j], feats2[j], feats3[j], feats4[j], feats5[j], feats6[j]])

        # compute differences
        d1 = y1mos[i] - y1mos[j]
        d2 = y2mos[i] - y2mos[j]
        d3 = y3mos[i] - y3mos[j]

        # concat targets
        targets = [y1mos[i], y2mos[i], y3mos[i], y1mos[j], y2mos[j], y3mos[j], d1, d2, d3]
        yield {
            "inp_pos": tf.cast(x0pos, dtype=tf.float32, name="inp_pos"),
            "inp_neg": tf.cast(x0neg, dtype=tf.float32, name="inp_neg"),
        }, tf.constant(targets, dtype=tf.float32, name="targets")


ds_valid = tf.data.Dataset.from_generator(
    lambda: generator_validationset(num_draws=65536),
    output_signature=(
        {
            "inp_pos": tf.TensorSpec(shape=(dim_features), dtype=tf.float32, name="inp_pos"),
            "inp_neg": tf.TensorSpec(shape=(dim_features), dtype=tf.float32, name="inp_neg"),
        },
        tf.TensorSpec(shape=(9), dtype=tf.float32, name="targets"),
    )
).batch(65536)

xy_valid = list(ds_valid)[0]  # generate once!


# Build the Scoring Model
def build_scoring_model(dim_features: int, 
                        n_units=64, activation="gelu", dropout=0.4):
    # the input tensor
    inputs = tf.keras.Input(shape=(dim_features,), name="inputs")

    # Dimensionality reduction layer
    x = tf.keras.layers.Dense(
        n_units,
        use_bias=False,
        kernel_initializer=tf.keras.initializers.VarianceScaling(seed=42),
        name='linear_reduce'
    )(inputs)
    x = tf.keras.layers.Activation(
        activation, name="linear_reduce_activation")(x)
    x = tf.keras.layers.Dropout(
        dropout, name="linear_reduce_dropout")(x)

    # Dense + 4.0
    out1_mos = tf.keras.layers.Dense(
        units=1, use_bias=False,
        kernel_initializer='glorot_uniform',
    )(x)
    out1_mos = tf.keras.layers.Lambda(
        lambda s: s + tf.constant(4.0), name='mos1'
    )(out1_mos)

    # Dense + 4.0
    out2_mos = tf.keras.layers.Dense(
        units=1, use_bias=False,
        kernel_initializer='glorot_uniform',
    )(x)
    out2_mos = tf.keras.layers.Lambda(
        lambda s: s + tf.constant(4.0), name='mos2'
    )(out2_mos)

    # Dense + 4.0
    out3_mos = tf.keras.layers.Dense(
        units=1, use_bias=False,
        kernel_initializer='glorot_uniform',
    )(x)
    out3_mos = tf.keras.layers.Lambda(
        lambda s: s + tf.constant(4.0), name='mos3'
    )(out3_mos)

    # Function API model
    model = tf.keras.Model(
        inputs=[inputs],
        outputs=[out1_mos, out2_mos, out3_mos, x],
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
    margin1 = tf.math.abs(y_true[:, 6]) / 6.0
    margin2 = tf.math.abs(y_true[:, 7]) / 6.0
    margin3 = tf.math.abs(y_true[:, 8]) / 6.0
    margin = tf.math.maximum(tf.math.maximum(margin1, margin2), margin3)
    # read model outputs
    n = (y_pred.shape[1] - 9) // 2
    repr_pos = y_pred[:, 9:(9 + n)]
    repr_neg = y_pred[:, (9 + n):(9 + n * 2)]
    # Triplet ranking loss between last representation layers
    dist = cosine_distance_normalized(repr_pos, repr_neg)
    loss = tf.reduce_mean(tf.math.maximum(0.0, margin - dist))
    return loss


def loss2_mse_diffs(y_true, y_pred):
    """ MSE loss between actual and predicted margins btw. pos & neg. ex """
    # norm to [0,1] by dividing by 6
    loss =  tf.reduce_mean(tf.math.pow(
        (y_true[:, 6] - (y_pred[:, 0] - y_pred[:, 3])) / 6.0, 2))
    loss += tf.reduce_mean(tf.math.pow(
        (y_true[:, 7] - (y_pred[:, 1] - y_pred[:, 4])) / 6.0, 2))
    loss += tf.reduce_mean(tf.math.pow(
        (y_true[:, 8] - (y_pred[:, 2] - y_pred[:, 5])) / 6.0, 2))
    return loss

def loss3_mse_target(y_true, y_pred):
    """ MSE loss on positive or negative noise targets """
    # norm to [0,1] by dividing by 6
    loss  = tf.reduce_mean(tf.math.pow((y_true[:, 0] - y_pred[:, 0]) / 6.0, 2))
    loss += tf.reduce_mean(tf.math.pow((y_true[:, 1] - y_pred[:, 1]) / 6.0, 2))
    loss += tf.reduce_mean(tf.math.pow((y_true[:, 2] - y_pred[:, 2]) / 6.0, 2))
    loss += tf.reduce_mean(tf.math.pow((y_true[:, 3] - y_pred[:, 3]) / 6.0, 2))
    loss += tf.reduce_mean(tf.math.pow((y_true[:, 4] - y_pred[:, 4]) / 6.0, 2))
    loss += tf.reduce_mean(tf.math.pow((y_true[:, 5] - y_pred[:, 5]) / 6.0, 2))
    return loss


def build_siamese_net(dim_features: int, 
                      n_units=64, activation="gelu", dropout=0.4):
    # the input tensors
    inp_pos = tf.keras.Input(shape=(dim_features,), name='inp_pos')
    inp_neg = tf.keras.Input(shape=(dim_features,), name='inp_neg')

    # linstantiate the shared scoring model
    scorer = build_scoring_model(
        dim_features=dim_features,
        n_units=n_units,
        activation=activation,
        dropout=dropout)

    # predict examples for each input
    y1_pos, y2_pos, y3_pos, repr_pos = scorer(inp_pos)
    y1_neg, y2_neg, y3_neg, repr_neg = scorer(inp_neg)

    # Function API model
    model = tf.keras.Model(
        inputs={
            'inp_pos': inp_pos,
            'inp_neg': inp_neg,
        },
        outputs=tf.keras.backend.concatenate([
            y1_pos, y2_pos, y3_pos,
            y1_neg, y2_neg, y3_neg,
            repr_pos, repr_neg
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
        loss=[loss1_rank_triplet, loss2_mse_diffs, loss3_mse_target],
        loss_weights=[0.5, 0.1, 0.4],
        metrics=[loss1_rank_triplet, loss2_mse_diffs, loss3_mse_target],
    )

    return model


# Training
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        min_delta=1e-6,
        patience=50,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=f"best-model-370b-siamese-{args.corr_trgt}",
        monitor="val_loss",
        mode="min",
        save_best_only=True,
    ),
]


model = build_siamese_net(
    dim_features, n_units=64, activation="gelu", dropout=0.4)


history = model.fit(
    ds_train, 
    validation_data=xy_valid,
    callbacks=callbacks,
    epochs=500,
)


with open(f"best-model-370b-siamese-{args.corr_trgt}/history.json", 'w') as fp:
    json.dump(history.history, fp)

