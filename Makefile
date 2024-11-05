install-requirements.global:
	pip install -r requirements.txt

pre-commit:
	pre-commit run --all-files

run-app:
	python gradio_ui.py

run-tool-isolated:
	python main.py
