# Variables
PYTHON_INTERPRETER = python3
VENV_NAME = .venv
VENV_ACTIVATE = $(VENV_NAME)/bin/activate
PIP = $(VENV_NAME)/bin/pip
PYTHON = $(VENV_NAME)/bin/python


# Create virtual environment
create: $(VENV_ACTIVATE)
$(VENV_ACTIVATE):
	$(PYTHON_INTERPRETER) -m venv $(VENV_NAME)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# Clean up
clean:
	rm -rf $(VENV_NAME)
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

clean_runs:
	rm -rf saved_models
	mkdir saved_models saved_models/mnist_models

.PHONY: all venv develop test clean
