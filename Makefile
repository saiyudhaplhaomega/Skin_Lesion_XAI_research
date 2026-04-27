# Research notebooks Makefile.

VENV_DIR := skin-lesion-env

ifeq ($(OS),Windows_NT)
	PYTHON      := py -3.13
	VENV_PYTHON := $(VENV_DIR)\Scripts\python.exe
	VENV_PIP    := $(VENV_DIR)\Scripts\pip.exe
else
	PYTHON      := python3.13
	VENV_PYTHON := $(VENV_DIR)/bin/python
	VENV_PIP    := $(VENV_DIR)/bin/pip
endif

.PHONY: help setup install install-dev register-kernel run-notebook train-backbones train-backbones-full train-checkpoints clean

help:
	@echo "Available targets:"
	@echo "  setup                - Create skin-lesion-env/ and install dev packages"
	@echo "  install              - Install production packages only (requirements.txt)"
	@echo "  install-dev          - Re-sync dev packages (requirements-dev.txt)"
	@echo "  register-kernel      - Register Jupyter kernel for VS Code"
	@echo "  run-notebook         - Launch Jupyter Lab in notebooks/"
	@echo "  train-backbones      - Train EfficientNet-B2 + MobileNetV2 (2 epochs, quick)"
	@echo "  train-backbones-full - Train EfficientNet-B2 + MobileNetV2 (15 epochs, full)"
	@echo "  train-checkpoints    - Train ResNet50 saving a checkpoint per epoch (for RQ5)"
	@echo "  clean                - Remove skin-lesion-env/ and __pycache__"

setup:
	@echo "Creating venv at $(VENV_DIR) ..."
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -r requirements-dev.txt
	@echo ""
	@echo "Done. Run: make register-kernel"

install:
	$(VENV_PYTHON) -m pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements.txt

install-dev:
	$(VENV_PYTHON) -m pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements-dev.txt

register-kernel:
	$(VENV_PYTHON) -m ipykernel install --user --name=skin-lesion-env --display-name="Skin Lesion (shared)"
	@echo "Kernel 'skin-lesion-env' registered."

run-notebook:
	$(VENV_PYTHON) -m jupyter lab --notebook-dir=./notebooks

train-backbones:
	$(VENV_PYTHON) train_backbones.py --epochs 2

train-backbones-full:
	$(VENV_PYTHON) train_backbones.py --epochs 15

train-checkpoints:
	$(VENV_PYTHON) train_epoch_checkpoints.py --epochs 10

clean:
	cmd /c "rmdir /s /q $(VENV_DIR)" 2>nul || rm -rf $(VENV_DIR)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
