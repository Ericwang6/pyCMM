#!/bin/bash
pip install torch==2.5.1+cpu torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cpu.html
pip install tensordict
pip install -e .
