install-requirements.global:
	pip install -r requirements.txt

run-app:
	python gradio_ui.py

run-tool-isolated:
	python main.py