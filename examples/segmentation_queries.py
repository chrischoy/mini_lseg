import os
import numpy as np
import argparse

from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch

from MiniLseg import init_lseg


def get_new_pallete(num_cls):
    n = num_cls
    pallete = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        pallete[j * 3 + 0] = 0
        pallete[j * 3 + 1] = 0
        pallete[j * 3 + 2] = 0
        i = 0
        while lab > 0:
            pallete[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            pallete[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            pallete[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            i = i + 1
            lab >>= 3
    return pallete


def get_new_mask_pallete(npimg, new_palette, out_label_flag=False, labels=None):
    """Get image color pallete for visualizing masks"""
    # put colormap
    out_img = Image.fromarray(npimg.squeeze().astype("uint8"))
    out_img.putpalette(new_palette)

    if out_label_flag:
        assert labels is not None
        u_index = np.unique(npimg)
        patches = []
        for i, index in enumerate(u_index):
            label = labels[index]
            cur_color = [
                new_palette[index * 3] / 255.0,
                new_palette[index * 3 + 1] / 255.0,
                new_palette[index * 3 + 2] / 255.0,
            ]
            red_patch = mpatches.Patch(color=cur_color, label=label)
            patches.append(red_patch)
    return out_img, patches


class Options:
    def __init__(self):
        parser = argparse.ArgumentParser(description="PyTorch Segmentation")
        # model
        parser.add_argument(
            "--model", type=str, default="encnet", help="model name (default: encnet)"
        )
        parser.add_argument(
            "--image_path",
            type=str,
            default=None,
        )
        parser.add_argument(
            "--device",
            type=str,
            default="cuda:0",
        )
        parser.add_argument(
            "--backbone",
            type=str,
            default="clip_vitl16_384",
            help="backbone name",
        )
        parser.add_argument("--max_size", type=int, default=520, help="max image size")
        parser.add_argument(
            "--crop_size", type=int, default=480, help="crop image size"
        )
        parser.add_argument(
            "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
        )
        # checking point
        parser.add_argument(
            "--weights", type=str, default="demo_e200.ckpt", help="checkpoint to test"
        )
        # evaluation option

        # test option
        parser.add_argument(
            "--no-scaleinv",
            dest="scale_inv",
            default=True,
            action="store_false",
            help="turn off scaleinv layers",
        )

        parser.add_argument(
            "--widehead", default=False, action="store_true", help="wider output head"
        )

        parser.add_argument(
            "--widehead_hr",
            default=False,
            action="store_true",
            help="wider output head",
        )
        parser.add_argument(
            "--ignore_index",
            type=int,
            default=-1,
            help="numeric value of ignore label in gt",
        )

        parser.add_argument(
            "--arch_option",
            type=int,
            default=0,
            help="which kind of architecture to be used",
        )

        parser.add_argument(
            "--block_depth",
            type=int,
            default=0,
            help="how many blocks should be used",
        )

        parser.add_argument(
            "--activation",
            choices=["lrelu", "tanh"],
            default="lrelu",
            help="use which activation to activate the block",
        )

        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        print(args)
        return args


if __name__ == "__main__":
    args = Options().parse()

    torch.manual_seed(args.seed)

    eval_fn, transform, eval_module = init_lseg(
        args.backbone,
        weight_path=args.weights,
        max_size=args.max_size,
        device=args.device,
    )

    assert os.path.exists(args.image_path)
    image = Image.open(args.image_path)
    pimage = transform(np.array(image)[..., :3]).to(args.device)

    while True:
        input_labels = input("labels (e.g. 'chair,tv,table,other'): ")
        labels = []
        for label in input_labels.split(","):
            labels.append(label.strip())

        outputs = eval_fn(pimage, labels)
        preds = [torch.max(output, 0)[1].cpu().numpy() for output in outputs]

        # Visualization
        palette = get_new_pallete(len(labels))
        mask, patches = get_new_mask_pallete(
            preds[0], palette, out_label_flag=True, labels=labels
        )
        seg = mask.convert("RGBA")
        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(image)
        plt.axis("off")
        plt.subplot(122)
        plt.imshow(seg)
        plt.legend(
            handles=patches,
            loc="upper right",
            bbox_to_anchor=(1.3, 1),
            prop={"size": 5},
        )
        plt.axis("off")
        plt.tight_layout()
        plt.show()
