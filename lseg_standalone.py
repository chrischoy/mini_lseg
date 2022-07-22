import os
import math
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import clip

from lseg_blocks import (
    Interpolate,
    _make_encoder,
    forward_vit,
    _make_fusion_block,
    depthwise_block,
    bottleneck_block,
)

up_kwargs = {"mode": "bilinear", "align_corners": True}


class LSeg(nn.Module):
    def __init__(
        self,
        head,
        features=256,
        backbone="clip_vitl16_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        **kwargs,
    ):
        super(LSeg, self).__init__()

        self.channels_last = channels_last

        hooks = {
            "clip_vitl16_384": [5, 11, 17, 23],
            "clipRN50x16_vitl16_384": [5, 11, 17, 23],
            "clip_vitb32_384": [2, 5, 8, 11],
        }

        # Instantiate backbone and reassemble blocks
        self.clip_pretrained, self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
        )

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
        if backbone in ["clipRN50x16_vitl16_384"]:
            self.out_c = 768
        else:
            self.out_c = 512
        self.scratch.head1 = nn.Conv2d(features, self.out_c, kernel_size=1)

        self.arch_option = kwargs["arch_option"]
        if self.arch_option == 1:
            self.scratch.head_block = bottleneck_block(activation=kwargs["activation"])
            self.block_depth = kwargs["block_depth"]
        elif self.arch_option == 2:
            self.scratch.head_block = depthwise_block(activation=kwargs["activation"])
            self.block_depth = kwargs["block_depth"]

        self.scratch.output_conv = head

        self.text = None if self.labels is None else clip.tokenize(self.labels)

    def forward(self, x, labelset=None):
        if labelset is None:
            text = self.text
        else:
            text = clip.tokenize(labelset)

        if self.channels_last:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        image_features = self.scratch.head1(path_1)

        imshape = image_features.shape
        image_features = image_features.permute(0, 2, 3, 1).reshape(-1, self.out_c)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        if text is None:
            out = image_features.view(imshape[0], imshape[2], imshape[3], -1).permute(
                0, 3, 1, 2
            )
            out = self.scratch.output_conv(out)
            return out

        text = text.to(x.device)
        text_features = self.clip_pretrained.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits_per_image = self.logit_scale * image_features.half() @ text_features.t()

        out = (
            logits_per_image.float()
            .view(imshape[0], imshape[2], imshape[3], -1)
            .permute(0, 3, 1, 2)
        )

        if self.arch_option in [1, 2]:
            for _ in range(self.block_depth - 1):
                out = self.scratch.head_block(out)
            out = self.scratch.head_block(out, False)

        out = self.scratch.output_conv(out)

        return out


class LSegNet(LSeg):
    """Network for semantic segmentation."""

    def __init__(
        self, labels=None, path=None, scale_factor=0.5, crop_size=480, **kwargs
    ):
        kwargs["use_bn"] = True

        self.crop_size = crop_size
        self.scale_factor = scale_factor
        self.labels = labels

        head = nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )

        super().__init__(head, **kwargs)


class LSegMultiEvalModule(nn.Module):
    """Multi-size Segmentation Eavluator"""

    def __init__(
        self,
        net,
        flip=True,
        scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
    ):
        super(LSegMultiEvalModule, self).__init__()
        self.base_size = 520
        self.crop_size = 480

        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]

        self._up_kwargs = up_kwargs
        self.scales = scales
        self.flip = flip
        self.net = net
        print(
            "MultiEvalModule: base_size {}, crop_size {}".format(
                self.base_size, self.crop_size
            )
        )

    def single_forward(self, image, label_set=None):
        batch, _, height, width = image.size()
        assert batch == 1
        self.nclass = 512 if label_set is None else len(label_set)
        stride_rate = 2.0 / 3.0
        crop_size = self.crop_size
        stride = int(crop_size * stride_rate)
        long_size, short_size = (height, width) if height > width else (width, height)

        if long_size <= crop_size:
            pad_img = pad_image(image, self.mean, self.std, crop_size)
            outputs = module_inference(self.net, pad_img, label_set, self.flip)
            outputs = crop_image(outputs, 0, height, 0, width)
        else:
            if short_size < crop_size:
                # pad if needed
                pad_img = pad_image(image, self.mean, self.std, crop_size)
            else:
                pad_img = image
            _, _, ph, pw = pad_img.shape  # .size()
            assert ph >= height and pw >= width
            # grid forward and normalize
            h_grids = int(math.ceil(1.0 * (ph - crop_size) / stride)) + 1
            w_grids = int(math.ceil(1.0 * (pw - crop_size) / stride)) + 1
            with torch.cuda.device_of(image):
                outputs = image.new().resize_(batch, self.nclass, ph, pw).zero_()
                count_norm = image.new().resize_(batch, 1, ph, pw).zero_()
            # grid evaluation
            for idh in range(h_grids):
                for idw in range(w_grids):
                    h0 = idh * stride
                    w0 = idw * stride
                    h1 = min(h0 + crop_size, ph)
                    w1 = min(w0 + crop_size, pw)
                    crop_img = crop_image(pad_img, h0, h1, w0, w1)
                    # pad if needed
                    pad_crop_img = pad_image(crop_img, self.mean, self.std, crop_size)
                    output = module_inference(
                        self.net, pad_crop_img, label_set, self.flip
                    )
                    outputs[:, :, h0:h1, w0:w1] += crop_image(
                        output, 0, h1 - h0, 0, w1 - w0
                    )
                    count_norm[:, :, h0:h1, w0:w1] += 1
            assert (count_norm == 0).sum() == 0
            outputs = outputs / count_norm
            outputs = outputs[:, :, :height, :width]
        return outputs

    def forward(self, image, label_set=None):
        """Mult-size Evaluation"""
        # only single image is supported for evaluation
        print("** MultiEvalModule forward phase: {} **".format(label_set))
        batch, _, h, w = image.size()
        assert batch == 1
        self.nclass = 512 if label_set is None else len(label_set)
        with torch.cuda.device_of(image):
            scores = image.new().resize_(batch, self.nclass, h, w).zero_()

        for scale in self.scales:
            long_size = int(math.ceil(self.base_size * scale))
            height, width, short_size = resize_hw_max(h, w, long_size)
            # resize image to current size
            cur_img = resize_image(image, height, width, **self._up_kwargs)
            outputs = self.single_forward(cur_img, label_set)
            score = resize_image(outputs, h, w, **self._up_kwargs)
            scores += score
        scores /= len(self.scales)
        return scores


def module_inference(module, image, label_set, flip=True):
    def get_pred(x):
        pred = module(x, label_set)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        return pred

    output = get_pred(image)
    if flip:
        fimg = flip_image(image)
        foutput = get_pred(fimg)
        output += flip_image(foutput)
        output /= 2
    return output


def resize_hw_max(h, w, long_size=520):
    if h > w:
        height = long_size
        width = int(1.0 * w * long_size / h + 0.5)
        short_size = width
    else:
        width = long_size
        height = int(1.0 * h * long_size / w + 0.5)
        short_size = height
    return height, width, short_size


def resize_image(img, h, w, **up_kwargs):
    return F.interpolate(img, (h, w), **up_kwargs)


def pad_image(img, mean, std, crop_size):
    b, c, h, w = img.shape  # .size()
    assert c == 3
    padh = crop_size - h if h < crop_size else 0
    padw = crop_size - w if w < crop_size else 0
    pad_values = -np.array(mean) / np.array(std)
    img_pad = img.new().resize_(b, c, h + padh, w + padw)
    for i in range(c):
        # note that pytorch pad params is in reversed orders
        img_pad[:, i, :, :] = F.pad(
            img[:, i, :, :], (0, padw, 0, padh), value=pad_values[i]
        )
    assert img_pad.size(2) >= crop_size and img_pad.size(3) >= crop_size
    return img_pad


def crop_image(img, h0, h1, w0, w1):
    return img[:, :, h0:h1, w0:w1]


def flip_image(img):
    assert img.dim() == 4
    with torch.cuda.device_of(img):
        idx = torch.arange(img.size(3) - 1, -1, -1).type_as(img).long()
    return img.index_select(3, idx)


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
            help="backbone name (default: resnet50)",
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


args = Options().parse()

torch.manual_seed(args.seed)
args.test_batch_size = 1
args.scale_inv = False
args.widehead = True
args.backbone = "clip_vitl16_384"
args.ignore_index = 255

net = LSegNet(
    backbone=args.backbone,
    num_features=256,
    aux_weight=0,
    se_loss=False,
    se_weight=0,
    base_lr=0,
    batch_size=1,
    max_epochs=0,
    ignore_index=args.ignore_index,
    dropout=0.0,
    scale_inv=args.scale_inv,
    augment=False,
    no_batchnorm=False,
    widehead=args.widehead,
    widehead_hr=args.widehead_hr,
    arch_option=0,
    block_depth=0,
    activation="lrelu",
)

eval_module = LSegMultiEvalModule(net)
weights = torch.load(args.weights, map_location=args.device)
eval_module.load_state_dict(weights["state_dict"])
eval_module = eval_module.eval()
eval_module = eval_module.to(args.device)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)
assert os.path.exists(args.image_path)
image = Image.open(args.image_path)
pimage = transform(np.array(image)[..., :3]).unsqueeze(0).to(args.device)
_, _, h, w = pimage.shape

# resize image to current size
if h > 520 or w > 520:
    height, width, _ = resize_hw_max(h, w, 520)
    pimage = resize_image(pimage, height, width, **up_kwargs)

# When label_set=None, generate the image features.
# Must use no_grad for small GPU memory usage
with torch.no_grad():
    output = eval_module(pimage)
    print(output.shape)

while True:
    input_labels = input("labels (e.g. 'chair,tv,table,other'): ")
    labels = []
    for label in input_labels.split(","):
        labels.append(label.strip())

    with torch.no_grad():
        outputs = eval_module(pimage, labels)
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
        handles=patches, loc="upper right", bbox_to_anchor=(1.3, 1), prop={"size": 5}
    )
    plt.axis("off")
    plt.tight_layout()
    plt.show()
