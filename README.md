# Mini Language Segmentation

This repository contains only essential components for inference of the language segmentation network ([lang-seg](https://github.com/isl-org/lang-seg)).

## Installation

First, install requirements and setup the package. Pretrained weights (`demo_e200.ckpt`) will be downloaded automatically.

```
pip install -r requirements.txt
python setup.py install
```

## Demo

```
python examples/segmentation_queries.py --image_path figs/outdoor.jpg --max_size 620
```

## Example output

![](./figs/Figure_1.png?raw=true)
