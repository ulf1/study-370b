import torch
import sentence_transformers as sbert
import trankit
import node_distance as nd
import epitran
import ipasymbols
import sfst_transduce
import re
import gc
from collections import Counter
import string
import numpy as np


# Load the pretrained models
hasgpu = torch.cuda.is_available()
model_sbert = sbert.SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
model_trankit = trankit.Pipeline(lang='german-hdt', gpu=hasgpu, cache_dir='./cache')
model_epi = epitran.Epitran('deu-Latn')
model_fst = sfst_transduce.Transducer("./SMOR/lib/smor.a")


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


def delete_models():
    # free memory
    del model_sbert
    del model_trankit
    del model_epi
    del model_fst
    gc.collect()
