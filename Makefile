.PHONY: install test lint manifests validate kb smoke mlflow

install:
	pip install -e ".[eval,dev]"

test:
	pytest -q

lint:
	ruff check src scripts tests

manifests:
	python scripts/create_data_manifests.py

validate:
	python scripts/validate_testset.py

kb:
	python scripts/build_kb.py --config configs/rag/rag_v1.yaml

smoke:
	python scripts/run_baseline.py --model configs/models/qwen3_8b.yaml --eval configs/evaluation/baseline_v1.yaml --limit 2 --no-judge --no-ragas

mlflow:
	mlflow ui --backend-store-uri ./mlruns --port 5000
