#!/bin/bash
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
pip install tensordict
pip install -e .
