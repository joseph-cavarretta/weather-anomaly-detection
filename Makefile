.PHONY: build train run

build:
	docker build -t weather-model .

train:
	docker run --rm \
		-v $(PWD)/src/data:/app/src/data \
		weather-model python src/train_model.py

run:
	docker run --rm \
		-v $(PWD)/src/data:/data \
		-v $(PWD)/src/data/scheduled_runs:/output \
		weather-model
