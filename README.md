# [370b] TextComplexityDE - Noise Targets 

## Install a virtual environment

```sh
python3.7 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

```sh
pip install "intel-tensorflow-avx512==2.8.0"
```

## Download dataset
```sh
wget -nc -q https://github.com/babaknaderi/TextComplexityDE/archive/refs/heads/master.zip
unzip -n master.zip
mv TextComplexityDE-master/TextComplexityDE19 data/
rm master.zip
rm -r TextComplexityDE-master
```

Download SMOR
```sh
wget -nc -q "https://www.cis.uni-muenchen.de/~schmid/tools/SMOR/data/SMOR-linux.zip"
unzip -n "SMOR-linux.zip"
rm SMOR-linux.zip
```

Download COW lemma frequencies
```sh
wget -nc -q "https://nlp-data-filestorage.s3.eu-central-1.amazonaws.com/word-frequencies/decow_wordfreq_cistem.csv.7z"
p7zip -d "decow_wordfreq_cistem.csv.7z"
mkdir -p decow
mv decow_wordfreq_cistem.csv decow/decow.csv
rm decow_wordfreq_cistem.csv.7z
```

Download DeReChar frequencies
```sh
wget -nc -q "https://www.ids-mannheim.de/fileadmin/kl/derewo/DeReChar-v-bi-DRC-2021-10-31-1.0.txt"
mkdir -p derechar
mv DeReChar-v-bi-DRC-2021-10-31-1.0.txt derechar/derechar.txt
```

## Preprocess Features

```sh
python 01-preprocess.py &
```

## Train models & infer
```
python3 11-train-rf.py && python3 11-infer.py
python3 16-train-rf.py && python3 16-infer.py
python3 17-train-rf.py && python3 17-infer.py
```
