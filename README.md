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

## Run experiments

```sh
nohup bash run-all.sh &
```
