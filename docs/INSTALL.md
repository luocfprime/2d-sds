# Installation

The project is packed as a python package.

```bash
conda env create -f environment.yml
```

Install PyTorch:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Install other dependencies:

```bash
pip install -e .
```

If there are dependency issues, you may refer to the `requirements.lock.txt`. It is tested with Python 3.10.9, PyTorch
2.0.1, CUDA 11.7.
