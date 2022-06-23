import torch
import sentence_transformers as sbert
import trankit
import node_distance as nd
import epitran
import ipasymbols
import sfst_transduce
import re
# import gc
from collections import Counter
import string
import numpy as np
import nltk
import pandas as pd
import gc


# Load the pretrained models
hasgpu = torch.cuda.is_available()
model_sbert = sbert.SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
model_trankit = trankit.Pipeline(lang='german-hdt', gpu=hasgpu, cache_dir='./cache')
model_epi = epitran.Epitran('deu-Latn')
model_fst = sfst_transduce.Transducer("./SMOR/lib/smor.a")

# load COW frequency list
stemmer = nltk.stem.Cistem()
df_decow = pd.read_csv('decow/decow.csv', index_col=['word'])
df_decow = df_decow[df_decow["freq"] > 100]  # removes ~97% of rows!
df_decow = np.log(df_decow + 1.)
df_decow = df_decow / df_decow.max()
# df_decow.at[stemmer.stem(word), 'freq']
decow = {row[0]: row[1].values[0] for row in df_decow.iterrows()}
del df_decow
gc.collect()

# load DeReChar frequency list
with open('derechar/derechar.txt', 'r') as fp:
    dat = fp.readlines()

dat = [s.lstrip() for s in dat]
dat = [s for s in dat if len(s) >= 2]
dat = [s for s in dat if 48 <= ord(s[0]) <= 57]
dat = [s.split(" ") for s in dat]
dat = [row for row in dat if len(row) == 2]
dat = [(int(num.replace(".", "")), bi.split("\n")[0]) for num, bi in dat]

derechar = {s: num for num, s in dat if len(s) == 1 and num > 0}
denom = max([num for _, num in derechar.items()])
derechar = {s: num / denom for s, num in derechar.items()}

derebigr = {s: num for num, s in dat if len(s) == 2 and num > 0}
denom = max([num for _, num in derebigr.items()])
derebigr = {s: num / denom for s, num in derebigr.items()}


# code for PoS-tag distribution
# https://universaldependencies.org/u/pos/
# -ndbt/nid- = not detected by trankit, or not in dataset
TAGSET = [
    'ADJ', 
    'ADP', 
    'ADV', 
    'AUX', 
    'CCONJ', 
    'DET', 
    'INTJ', 
    'NOUN', 
    'NUM',
    'PART', 
    'PRON', 
    'PROPN', 
    'PUNCT', 
    'SCONJ', 
    # 'SYM',  # -ndbt/nid-
    'VERB', 
    'X'
]


def get_postag_distribution(postags):
    n_tokens = len(postags)
    cnt = Counter(postags)
    pdf = np.zeros((len(TAGSET),))
    for key, val in cnt.items():
        try:
            idx = TAGSET.index(key)
        except:
            idx = TAGSET.index("X")
            # idx = len(TAGSET)
        pdf[idx] = val / n_tokens
    return pdf


# morphtags
# Ensure that all STTS conversions are included
# https://universaldependencies.org/tagset-conversion/de-stts-uposf.html
# -ndbt/nid- = not detected by trankit, or not in dataset
MORPHTAGS = [
    # punctuation type (3/11)
    "PunctType=Brck",  # `$(`
    "PunctType=Comm",  # `$,`
    "PunctType=Peri",  # `$.`
    # adposition type (3/4)
    "AdpType=Post",  # APPO
    "AdpType=Prep",  # APPR, APPRART
    "AdpType=Circ",  # APZR
    # particle type (3/6)
    "PartType=Res",  # PTKANT
    "PartType=Vbp",  # PTKVZ
    "PartType=Inf",  # PTKZU
    # pronominal type (8/11)
    "PronType=Art",  # APPRART, ART
    "PronType=Dem",  # PAV, PDAT, PDS
    "PronType=Ind",  # PIAT, PIDAT, PIS
    # "PronType=Neg",  # PIAT, PIDAT, PIS -ndbt/nid-
    # "PronType=Tot",  # PIAT, PIDAT, PIS -ndbt/nid-
    "PronType=Prs",  # PPER, PPOSAT, PPOSS, PRF
    "PronType=Rel",  # PRELAT, PRELS
    "PronType=Int",  # PWAT, PWAV, PWS
    # other related to STTS post tags
    # "AdjType=Pdt",  # PIDAT -ndbt/nid-
    "ConjType=Comp",  # KOKOM
    "Foreign=Yes",  # FM
    "Hyph=Yes",  # TRUNC
    "NumType=Card",  # CARD
    "Polarity=Neg",  # PTKNEG
    "Poss=Yes",  # PPOSAT, PPOSS
    "Reflex=Yes",  # PRF
    "Variant=Short",  # ADJD
    # verbs
    "VerbForm=Fin",  # VAFIN, VAIMP, VMFIN, VVFIN, VVIMP
    "VerbForm=Inf",  # VAINF, VVINF, VVIZU
    "VerbForm=Part",  # VAPP, VMPP, VVPP
    "Mood=Ind",  # VAFIN, VMFIN, VVFIN
    "Mood=Imp",  # VAIMP, VVIMP
    # "Mood=Sub",  # -ndbt/nid-
    "Aspect=Perf",  # VAPP, VMPP, VVPP
    "VerbType=Mod",  # VMPP
    # other syntax
    "Gender=Fem", 
    "Gender=Masc", 
    "Gender=Neut",
    "Number=Sing", 
    "Number=Plur",
    "Person=1", 
    "Person=2", 
    "Person=3",
    "Case=Nom", 
    "Case=Dat", 
    "Case=Gen", 
    "Case=Acc",
    # "Definite=Ind",  # -ndbt/nid-
    # "Definite=Def",  # -ndbt/nid-
    "Degree=Pos", 
    "Degree=Cmp", 
    "Degree=Sup",
    "Tense=Pres", 
    "Tense=Past", 
    # "Tense=Fut",  # -ndbt/nid-
    # "Tense=Imp",  # -ndbt/nid-
    # "Tense=Pqp",  # -ndbt/nid-
    # "Polite=",  # -ndbt/nid-  
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


def get_word_freq_distribution(sent):
    words = sent.split(" ")
    freqs = [decow.get(stemmer.stem(w), 0) for w in words]
    pdf = np.histogram(freqs, bins=6, range=(0.0, 1.0))[0] / len(freqs)
    return pdf


def get_char_bigram_freq_metrics(txt):
    metrics = [0., 0., 0., 0.]
    # average char freq
    freq = [derechar.get(c, 0.0) for c in txt]
    metrics[0] = sum(freq) / len(txt)
    # avg freq of 50% least frequent chars
    if len(freq) >= 2:
        least = sorted(freq)[:len(freq) // 2]
        metrics[1] = sum(least) / len(least)
    # average bigram freq
    if len(txt) >= 3:
        freq = [derebigr.get(txt[i:(i + 2)], 0.0) 
                for i in range(1, len(txt) - 1)]
        metrics[2] = sum(freq) / len(freq)
        # avg freq of 50% least frequent chars
        if len(freq) >= 2:
            least = sorted(freq)[:len(freq) // 2]
            metrics[3] = sum(least) / len(least) 
    # done
    return metrics


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
        postags = [t.get("upos") for t in snt['tokens']]
        pdf = get_postag_distribution(postags)
        feats3.append(pdf)
        # Morphological attributes
        pdf = get_morphtag_distribution(snt)
        feats4.append(pdf)

    # other tools
    feats5 = []
    feats6 = []  # SMOR: Num morphemes divded by number of words
    feats7 = []  # COW lemma frequencies
    feats8 = []  # sentence length
    feats9 = []  # char/bi-gram frequency
    for txt in texts:
        # phonetics, consonant clusters
        pdf = get_consonant_distibution(txt)
        feats5.append(pdf)
        # morphology/lexeme stats
        pdf1, pdf2, pdf3 = get_morphology_distributions(txt)
        feats6.append(np.hstack([pdf1, pdf2, pdf3]))
        # lemma frequencies
        pdf = get_word_freq_distribution(txt)
        feats7.append(pdf)
        # sentence length
        words = txt.split(" ")
        feats8.append(np.log([len(words) + 1., len(txt) + 1.]))
        # char/bigram frequencies
        metrics = get_char_bigram_freq_metrics(txt)
        feats9.append(metrics)

    # done
    return (
        feats1, np.vstack(feats2), np.vstack(feats3),
        np.vstack(feats4), np.vstack(feats5), np.vstack(feats6),
        np.vstack(feats7), np.vstack(feats8), np.array(feats9)
    )
