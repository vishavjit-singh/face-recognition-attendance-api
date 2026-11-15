PY ?= python3
export PYTHONPATH := src

.PHONY: install models demo report summary evaluate serve test lint up down clean

install:
	$(PY) -m pip install -r requirements-dev.txt

## Download YuNet + SFace (OpenCV Zoo) and verify the pinned SHA-256
models:
	$(PY) -m attendance models

## Real-model plumbing check + one synthetic day through the API → charts → summary
demo: models
	$(PY) -m attendance demo
	$(PY) -m attendance report
	$(PY) -m attendance summary

report:
	$(PY) -m attendance report

summary:
	$(PY) -m attendance summary

## Accuracy on the Olivetti research faces (scikit-learn downloads them; results in reports/)
evaluate: models
	$(PY) -m attendance evaluate

serve: models
	$(PY) -m attendance serve

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .

up:
	docker compose up --build

down:
	docker compose down

clean:
	rm -rf data reports
