.PHONY: data prep train tune register pipeline flow

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

flow:
	DO_NOT_TRACK=1 PREFECT_SERVER_ANALYTICS_ENABLED=false python -m src.flow

export:
	python -m src.export_model

docker-build: export
	docker build -t term-deposit-api:0.1 .

docker-run:
	docker run --rm -p 8000:8000 term-deposit-api:0.1

batch:
	python -m src.batch_score --input data/raw/bank-full.csv --sep ";" --output data/scored/call_list.csv