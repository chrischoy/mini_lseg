# Standalone Language Segmentation

This repository contains a standalone language segmentation network from [lang-seg](https://github.com/isl-org/lang-seg).
The original repository contains a lot of legacy packages that do not work with the latest pytorch. This repo contains the minimal codebase for inference only.

To download the ViT-L/16 weight with the CLIP ViT-B/32 text encoder, go to [url](https://drive.google.com/file/d/1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb/view?usp=sharing) and download the `demo_e200.ckpt`.

```
python lseg.py --image_path XXX.jpg --weights demo_e200.ckpt
```

## Example output

![](./figs/Figure_1.png?raw=true)
