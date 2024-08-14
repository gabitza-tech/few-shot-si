#!/bin/bash

export PYTHONPATH=$(pwd)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
