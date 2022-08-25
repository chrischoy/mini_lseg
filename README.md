# Standalone Language Segmentation

This repository contains a standalone language segmentation network from [lang-seg](https://github.com/isl-org/lang-seg).
The original repository contains a lot of legacy packages that do not work with the latest pytorch. This repo contains the minimal codebase for inference only.

## Installation

First, install requirements and setup the package. Then download the pretrained weights.

```
pip install -r requirements.txt
python setup.py install
```

Download the ViT-L/16 weight with the CLIP ViT-B/32 text encoder.
Go to the [url](https://drive.google.com/file/d/1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb/view?usp=sharing) and download the `demo_e200.ckpt`.

## Demo

```
python examples/segmentation_queries.py --image_path figs/outdoor.jpg --weights demo_e200.ckpt --max_size 620
```

## Example output

![](./figs/Figure_1.png?raw=true)
