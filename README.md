# Swin ImageNet

Training of Swin Transformer models on ImageNEt using the Monai library

## Installation

Make sure Kaggle credential are at ~/.kaggle/kaggle.json

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python prepare_data.py --competition imagenet-object-localization-challenge --dest /data/imagenet --keep-archives

python train_swin.py --data-dir /data/imagenet --variant tiny --input-size 224 --batch-size 256 --epochs 90 --workers 8 --output-dir ./checkpoints