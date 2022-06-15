#!/bin/bash

source .venv/bin/activate

python3 script.py --corr-trgt=0 --corr-regul=0
python3 script.py --corr-trgt=1 --corr-regul=1
python3 script.py --corr-trgt=1 --corr-regul=0

python3 script2.py --corr-trgt=0 --corr-regul=0
python3 script2.py --corr-trgt=1 --corr-regul=1
python3 script2.py --corr-trgt=1 --corr-regul=0

python3 script3.py --corr-trgt=0 --corr-regul=0
python3 script3.py --corr-trgt=1 --corr-regul=1
python3 script3.py --corr-trgt=1 --corr-regul=0
python3 script3.py --corr-trgt=0 --corr-regul=1
