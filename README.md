# Standalone Language Segmentation

This repository contains a standalone language segmentation network from [lang-seg](https://github.com/isl-org/lang-seg).
The original repository contains a lot of legacy packages that do not work with the latest pytorch. This repo contains the minimal codebase for inference only.

## Installation

First, install requirements and setup the package. Pretrained weights will be downloaded automatically.

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
