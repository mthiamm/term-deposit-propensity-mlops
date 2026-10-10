.PHONY: data prep train tune register pipeline

data:
	wget -P data/raw "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
	unzip -o data/raw/bank+marketing.zip -d data/raw/
	unzip -o data/raw/bank.zip -d data/raw/

prep:
	python src/data_prep.py

train:
	python -m src.train --model hgb

tune:
	python -m src.tune --trials 30

register:
	python -m src.register

pipeline: prep train tune register