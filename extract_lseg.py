import os
import gin
import numpy as np
import argparse
from PIL import Image

import torch
from lseg import resize_hw_max, resize_image, up_kwargs

GLOBAL_LSEG = {}


@torch.no_grad()
def init_lseg(
    backbone="clip_vitl16_384",
    weight_path=None,
    max_size=320,
    device="cuda",
):
    global GLOBAL_LSEG
    if "eval_module" not in GLOBAL_LSEG:
        eval_module, transform = init_lseg(
            args.backbone,
            weight_path=weight_path,
            max_size=max_size,
            device=device,
        )
        GLOBAL_LSEG["eval_module"] = eval_module
        GLOBAL_LSEG["transform"] = transform
    else:
        eval_module, transform = GLOBAL_LSEG["eval_module"], GLOBAL_LSEG["transform"]
    return eval_module, transform


@gin.configurable
@torch.no_grad()
def extract_and_save_lseg(
    image_path,
    max_size=320,
    weight_path=None,
    lseg_suffix="lseg",
    device="cuda",
):
    eval_module, transform = init_lseg(
        weight_path=weight_path, max_size=max_size, device=device
    )

    assert os.path.exists(image_path)
    image = Image.open(image_path)
    pimage = transform(np.array(image)[..., :3]).unsqueeze(0).to(device)
    outputs = eval_module(pimage)
    file_name = os.path.basename(image_path)
    file_name = file_name.split(".")[0]
    dir_name = os.path.dirname(image_path)
    torch.save(
        outputs.half().cpu(),
        os.path.join(dir_name, "/", f"{file_name}_{lseg_suffix}.th"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--ginb",
        action="append",
        help="gin bindings",
    )
    args = parser.parse_args()
    print(f"Gin bindings: {args.ginb}")
    # Seed
    gin.parse_config_files_and_bindings([], args.ginb)

    extract_and_save_lseg(
        "~/datasets/nerf_llff_data/trex/images/DJI_20200223_163548_810.jpg"
    )
