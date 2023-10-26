# Variables
PYTHON_INTERPRETER = python3
VENV_NAME = .venv
VENV_ACTIVATE = $(VENV_NAME)/bin/activate
PIP = $(VENV_NAME)/bin/pip
PYTHON = $(VENV_NAME)/bin/python

# Default target
all: venv

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

.PHONY: all venv develop test clean
