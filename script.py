import pandas as pd
import numpy as np
import sentence_transformers as sbert
import trankit
import node_distance as nd
import ipasymbols
import epitran
import sfst_transduce
import re
import matplotlib.pyplot as plt
import torch
import tensorflow as tf
import json
import gc
from collections import Counter
import string
import re
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--corr-trgt', type=int, default=0)
args = parser.parse_args()

# Load the pretrained models
hasgpu = torch.cuda.is_available()
model_sbert = sbert.SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
model_trankit = trankit.Pipeline(lang='german-hdt', gpu=hasgpu, cache_dir='./cache')
model_epi = epitran.Epitran('deu-Latn')
model_fst = sfst_transduce.Transducer("./SMOR/lib/smor.a")


# Load the raw data
df = pd.read_csv("data/ratings.csv", encoding="ISO-8859-1")
# print(df.columns)


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


# Feature Engineering

# code for PoS-tag distribution
TAGSET = [
    '$(', '$,', '$.', 'ADJA', 'ADJD', 'ADV', 'APPO', 'APPR', 'APPRART',
    'APZR', 'ART', 'CARD', 'FM', 'ITJ', 'KOKOM', 'KON', 'KOUI', 'KOUS',
    'NE', 'NN', 'NNE', 'PDAT', 'PDS', 'PIAT', 'PIS', 'PPER', 'PPOSAT',
    'PPOSS', 'PRELAT', 'PRELS', 'PRF', 'PROAV', 'PTKA', 'PTKANT',
    'PTKNEG', 'PTKVZ', 'PTKZU', 'PWAT', 'PWAV', 'PWS', 'TRUNC', 'VAFIN',
    'VAIMP', 'VAINF', 'VAPP', 'VMFIN', 'VMINF', 'VMPP', 'VVFIN', 'VVIMP',
    'VVINF', 'VVIZU', 'VVPP', 'XY', '_SP']

def get_postag_distribution(postags):
    n_tokens = len(postags)
    cnt = Counter(postags)
    pdf = np.zeros((len(TAGSET) + 1,))
    for key, val in cnt.items():
        try:
            idx = TAGSET.index(key)
        except:
            idx = len(TAGSET)
        pdf[idx] = val / n_tokens
    return pdf


# morphtags
MORPHTAGS = [
    "Gender=Fem", "Gender=Masc", "Gender=Neut",
    "Number=Sing", "Number=Plur",
    "Person=1", "Person=2", "Person=3",
    "Case=Nom", "Case=Dat", "Case=Gen", "Case=Acc",
    "Definite=Ind", "Definite=Def",
    "VerbForm=Conv", "Verbform=Fin", "Verbform=Gdv", "Verbform=Ger", 
    "Verbform=Inf", "Verbform=Part", "Verbform=Sup", "Verbform=Vnoun",
    "Degree=Pos", "Degree=Cmp", "Degree=Sup",
    "Polarity=Neg",
    "Mood=Ind", "Mood=Imp", "Mood=Sub",
    "Tense=Pres", "Tense=Past",
    "NumType=",  # any
    "Poss=",
    "Reflex=",
    "Polite="
]

def get_morphtag_distribution(snt):
    n_token = len(snt['tokens'])
    cnt = np.zeros((len(MORPHTAGS),))
    for t in snt['tokens']:
        mfeats = t.get("feats")
        if isinstance(mfeats, str):
            for idx, tag in enumerate(MORPHTAGS):
                if tag in mfeats:
                    cnt[idx] += 1
    return cnt / n_token


# https://github.com/linguistik/ipasymbols/blob/main/ipasymbols/utils.py#L56
def get_consonant_distibution(txt):
    ipatxt = model_epi.transliterate(txt)
    # identify clusters of 2 and 3, and count consonants
    types = ["pulmonic", "non-pulmonic", "affricate", "co-articulated"]
    clusters = ipasymbols.count_clusters(
        ipatxt, query={"type": types}, 
        phonlen=3, min_cluster_len=1)
    # compute ratios
    n_len = len(ipatxt)
    return np.array([clusters.get(1, 0.0), clusters.get(2, 0.0), clusters.get(3, 0.0)]) / n_len


# Morphemes/Lexemes
def get_morphology_stats(word):
    res = model_fst.analyse(word)
    if len(res) == 0:
        return 0, 0, 0
    variants = {}
    for sinp in res:
        s = re.sub(r'<[^>]*>', '\t', sinp)
        lexemes = [t for t in s.split("\t") if len(t) > 0]
        key = "+".join(lexemes)
        variants[key] = lexemes
    num_usecases = len(res)  # syntactial ambivalence
    num_splittings = len(variants)  # lexeme ambivalence
    max_lexemes = max([len(lexemes) for lexemes in variants.values()])  # working memory for composita comprehension
    return num_usecases, num_splittings, max_lexemes

def simple_tokenizer(sinp):
    chars = re.escape(string.punctuation)
    s = re.sub(r'['+chars+' ]', '\t', sinp)
    tokens = [re.sub('[^a-zA-Z0-9]', '', t) for t in s.split('\t')]
    tokens = [t for t in tokens if len(t) > 0]
    return tokens

def get_morphology_distributions(sent):
    tokens = simple_tokenizer(sent)
    # get stats
    num_usecases = []
    num_splittings = []
    max_lexemes = []
    for word in tokens:
        tmp = get_morphology_stats(word)
        num_usecases.append(tmp[0])
        num_splittings.append(tmp[1])
        max_lexemes.append(tmp[2])
    # convert to empirical pdf
    # (A) syntactial ambivalence
    n_tokens = len(tokens)
    cnt1 = Counter(num_usecases)
    pdf1 = np.zeros((12,))
    for key, val in cnt1.items():
        idx = min(max(0, key - 1), 11)
        pdf1[idx] = val / n_tokens
    # (B) lexeme ambivalence
    cnt2 = Counter(num_splittings)
    pdf2 = np.zeros((4,))
    for key, val in cnt2.items():
        idx = min(max(0, key - 1), 3)
        pdf2[idx] = val / n_tokens
    # (C) working memory for composita comprehension
    cnt3 = Counter(max_lexemes)
    pdf3 = np.zeros((4,))
    for key, val in cnt3.items():
        idx = min(max(0, key - 1), 3)
        pdf3[idx] = val / n_tokens
    # done
    return pdf1, pdf2, pdf3


# transfomer function
def preprocessing(texts):
    # semantic
    feats1 = model_sbert.encode(texts)

    # trankit
    feats2 = []
    feats3 = []
    feats4 = []
    for txt in texts:
        # parse sentence
        snt = model_trankit(txt, is_sent=True)
        # node vs token distance
        edges = [(t.get("head"), t.get("id"))
                for t in snt.get("tokens")
                if isinstance(t.get("id"), int)]
        num_nodes = len(snt.get("tokens")) + 1
        nodedist, tokendist, indicies = nd.node_token_distances(
            [edges], [num_nodes], cutoff=25)
        _, pdf, _ = nd.tokenvsnode_distribution(
            tokendist, nodedist, xmin=-5, xmax=15)
        feats2.append(pdf)
        # PoS tag distribution
        postags = [t.get("xpos") for t in snt['tokens']]
        pdf = get_postag_distribution(postags)
        feats3.append(pdf)
        # Morphological attributes
        pdf = get_morphtag_distribution(snt)
        feats4.append(pdf)

    # other tools
    feats5 = []
    feats6 = []  # SMOR: Num morphemes divded by number of words
    for txt in texts:
        # phonetics, consonant clusters
        pdf = get_consonant_distibution(txt)
        feats5.append(pdf)
        # morphology/lexeme stats
        pdf1, pdf2, pdf3 = get_morphology_distributions(txt)
        feats6.append(np.hstack([pdf1, pdf2, pdf3]))

    # done
    return feats1, feats2, feats3, feats4, feats5, feats6


# preprocess all examples
feats1, feats2, feats3, feats4, feats5, feats6 = preprocessing(texts)


print("Number of features")
print(f"{'Num SBert':>20s}: {feats1[0].shape[0]}")
print(f"{'Node vs Token dist.':>20s}: {feats2[0].shape[0]}")
print(f"{'PoS tag distr.':>20s}: {feats3[0].shape[0]}")
print(f"{'Morph. tags':>20s}: {feats4[0].shape[0]}")
print(f"{'Consonant cluster':>20s}: {feats5[0].shape[0]}")
print(f"{'Morph./lexemes':>20s}: {feats6[0].shape[0]}")


# free memory
del model_sbert
del model_trankit
del model_epi
del model_fst
gc.collect()


# Training set
def generator_trainingset(num_bootstrap: int = 64):
    num_data = len(feats1)
    for j in range(num_bootstrap):
        for i in range(num_data):
            # merge features
            xinputs = np.hstack([feats1[i], feats2[i], feats3[i], feats4[i], feats5[i], feats6[i]])
            
            if args.corr_trgt == 0:
                # simulate uncorrelated random scores
                y1 = np.random.normal(loc=y1mos[i], scale=y1std[i], size=1)
                y2 = np.random.normal(loc=y2mos[i], scale=y2std[i], size=1)
                y3 = np.random.normal(loc=y3mos[i], scale=y3std[i], size=1)
                y1 = np.maximum(1.0, np.minimum(7.0, y1))[0]
                y2 = np.maximum(1.0, np.minimum(7.0, y2))[0]
                y3 = np.maximum(1.0, np.minimum(7.0, y3))[0]
            elif args.corr_trgt == 1:
                # simulate correlated random scores
                Y = np.random.standard_normal((1, 3))
                Y = np.dot(Y, np.linalg.cholesky(y_rho).T)[0]
                y1 = np.maximum(1.0, np.minimum(7.0, Y[0] * y1std[i] + y1mos[i]))
                y2 = np.maximum(1.0, np.minimum(7.0, Y[1] * y2std[i] + y2mos[i]))
                y3 = np.maximum(1.0, np.minimum(7.0, Y[2] * y3std[i] + y3mos[i]))

            yield {
                "inputs": tf.cast(xinputs, dtype=tf.float32, name="inputs")
            }, {
                "out1_mos": tf.cast(y1, dtype=tf.float32, name="out1_mos"),
                "out2_mos": tf.cast(y2, dtype=tf.float32, name="out2_mos"),
                "out3_mos": tf.cast(y3, dtype=tf.float32, name="out3_mos"),
            }

batch_size = 128

dim_features = len(feats1[0]) + len(feats2[0]) + len(feats3[0]) + len(feats4[0]) + len(feats5[0]) + len(feats6[0])


ds_train = tf.data.Dataset.from_generator(
    lambda: generator_trainingset(
        num_bootstrap=64
    ),
    output_signature=(
        {
            "inputs": tf.TensorSpec(shape=(dim_features), dtype=tf.float32, name="inputs")
        },
        {
            "out1_mos": tf.TensorSpec(shape=(), dtype=tf.float32, name="out1_mos"),
            "out2_mos": tf.TensorSpec(shape=(), dtype=tf.float32, name="out2_mos"),
            "out3_mos": tf.TensorSpec(shape=(), dtype=tf.float32, name="out3_mos"),
        }
    )
).batch(batch_size)


# Validation set
def generator_validationset():
    num_data = len(feats1)
    for i in range(num_data):
        xinputs = np.hstack([feats1[i], feats2[i], feats3[i], feats4[i], feats5[i], feats6[i]])
        yield {
            "inputs": tf.cast(xinputs, dtype=tf.float32, name="inputs"),
        }, {
            "out1_mos": tf.cast(y1mos[i], dtype=tf.float32, name="out1_mos"),
            "out2_mos": tf.cast(y2mos[i], dtype=tf.float32, name="out2_mos"),
            "out3_mos": tf.cast(y3mos[i], dtype=tf.float32, name="out3_mos"),
        }

ds_valid = tf.data.Dataset.from_generator(
    lambda: generator_validationset(),
    output_signature=(
        {
            "inputs": tf.TensorSpec(shape=(dim_features), dtype=tf.float32, name="inputs"),
        },
        {
            "out1_mos": tf.TensorSpec(shape=(), dtype=tf.float32, name="out1_mos"),
            "out2_mos": tf.TensorSpec(shape=(), dtype=tf.float32, name="out2_mos"),
            "out3_mos": tf.TensorSpec(shape=(), dtype=tf.float32, name="out3_mos"),
        }
    )
).batch(batch_size)


# Modeling

# the input tensor
inputs = tf.keras.Input(shape=(dim_features,), name="inputs")

# Dense + 4.0
out1_mos = tf.keras.layers.Dense(
    units=1, use_bias=False,
    kernel_initializer='glorot_uniform',
)(inputs)
out1_mos = tf.keras.layers.Lambda(
    lambda s: s + tf.constant(4.0), name='mos1'
)(out1_mos)

# Dense + 4.0
out2_mos = tf.keras.layers.Dense(
    units=1, use_bias=False,
    kernel_initializer='glorot_uniform',
)(inputs)
out2_mos = tf.keras.layers.Lambda(
    lambda s: s + tf.constant(4.0), name='mos2'
)(out2_mos)

# Dense + 4.0
out3_mos = tf.keras.layers.Dense(
    units=1, use_bias=False,
    kernel_initializer='glorot_uniform',
)(inputs)
out3_mos = tf.keras.layers.Lambda(
    lambda s: s + tf.constant(4.0), name='mos3'
)(out3_mos)

# Function API model
model = tf.keras.Model(
    inputs=[inputs],
    outputs={
        "out1_mos": out1_mos, 
        "out2_mos": out2_mos,
        "out3_mos": out3_mos,
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
    loss={
        "out1_mos": "mean_squared_error",
        "out2_mos": "mean_squared_error",
        "out3_mos": "mean_squared_error",
    },
)

history = model.fit(
    ds_train, 
    validation_data=ds_valid,
    callbacks=callbacks,
    epochs=500,
)

with open(f"best-model-370b-{args.corr_trgt}/history.json", 'w') as fp:
    json.dump(history.history, fp)

