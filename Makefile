.PHONY: run dev test eval tune clean install

install:
	pip install -r requirements.txt

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

dev:
	uvicorn app.main:app --reload --port 8000

test:
	python -m pytest tests/test_unit.py -v

eval:
	python tests/eval.py

tune:
	python tests/tune.py

clean:
	rm -f nexusai.db
	rm -f tests/results.json tests/report.md tests/confusion_matrix.png
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

docker:
	docker compose up --build
