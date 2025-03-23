import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import argparse
import time
import torch
import re
import torch.nn as nn
from torch.utils.data import DataLoader
import random
from PIL import Image, ImageFilter, ImageOps
import timm
from torchvision.transforms import transforms
from tqdm import tqdm
import numpy as np
from datasets.cls_dataset import cls_dataset


def recreate_image(im_as_var):
    """
    Recreates images from a torch variable for a batch.
    """
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1 / 0.229, 1 / 0.224, 1 / 0.225]

    recreated_images = []
    for img in im_as_var:
        recreated_im = img.data.numpy().copy()
        for c in range(3):
            recreated_im[c] /= reverse_std[c]
            recreated_im[c] -= reverse_mean[c]
        recreated_im[recreated_im > 1] = 1
        recreated_im[recreated_im < 0] = 0
        recreated_im = np.uint8(np.round(recreated_im * 255)).transpose(1, 2, 0)
        recreated_images.append(recreated_im)

    return recreated_images


def preprocess_image_batch_gpu(image_list, mean=None, std=None):
    """
    Preprocesses a batch of images to tensors directly on GPU.
    Args:
        image_list: List of images as NumPy arrays.
        mean: List of mean values for normalization.
        std: List of standard deviation values for normalization.
    Returns:
        A torch.Tensor of preprocessed images with shape [batch_size, channels, crop_size, crop_size].
    """
    if std is None:
        std = [0.229, 0.224, 0.225]
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    image_array = np.stack(image_list)  # shape: [batch_size, height, width, channels]

    image_batch = torch.tensor(image_array, dtype=torch.float32, device='cuda').permute(0, 3, 1, 2) / 255.0

    mean_tensor = torch.tensor(mean, device='cuda').view(1, -1, 1, 1)
    std_tensor = torch.tensor(std, device='cuda').view(1, -1, 1, 1)
    image_batch = (image_batch - mean_tensor) / std_tensor

    return image_batch


class TwoCropsTransform:
    """Take two random crops of one image"""

    def __init__(self, base_transform1, base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

    def __call__(self, x):
        im1 = self.base_transform1(x)
        im2 = self.base_transform2(x)
        return [im1, im2]


class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def __call__(self, x):
        return ImageOps.solarize(x)


class CLARS(nn.Module):
    def __init__(self, backbone, pretrained_path=None, target_layer=None, num_classes=45):
        """
        Args:
            backbone (str): Backbone model name.
            pretrained (bool): Whether to load pretrained weights.
            target_layer (str): The name of the target layer to extract features.
            num_classes (int): The number of classes for classification.
        """
        super(CLARS, self).__init__()
        if backbone == 'rsmamba_s':
            from models.rsmamba import RSMamba
            self.encoder = RSMamba(arch='small', out_type='avg_featmap', img_size=224, num_classes=num_classes,
                                   patch_cfg=dict(stride=8),
                                   init_cfg=[dict(type='Kaiming', layer='Conv2d', mode='fan_in', nonlinearity='linear')]
                                   )
        elif backbone == 'rsmamba_b':
            from models.rsmamba import RSMamba
            self.encoder = RSMamba(arch='b', out_type='avg_featmap', img_size=224, num_classes=num_classes,
                                   patch_cfg=dict(stride=8),
                                   init_cfg=[dict(type='Kaiming', layer='Conv2d', mode='fan_in', nonlinearity='linear')]
                                   )
        elif backbone == 'replknet':
            from models.replknet import create_RepLKNet31B
            self.encoder = create_RepLKNet31B(num_classes=num_classes, small_kernel_merged=False, use_checkpoint=False)
        else:
            self.encoder = timm.create_model(backbone, pretrained=True, num_classes=num_classes)
        if pretrained_path:
            self.encoder.load_state_dict(torch.load(pretrained_path))
            print(f"Load pretrained model from {pretrained_path}")
        self.target_layer = target_layer
        self.feature = None

        # Register hook for the target layer
        if self.target_layer:
            layer = dict([*self.encoder.named_modules()]).get(target_layer, None)
            if layer:
                layer.register_forward_hook(self._hook_fn)
            else:
                raise ValueError(f"Layer {target_layer} not found in the model.")

    def _hook_fn(self, module, input, output):
        self.feature = output

    def forward(self, x):
        # Forward pass for x1
        self.feature = None
        logits = self.encoder(x)  # Get the logits for x1
        feature = self.feature  # Store the feature captured by the hook

        return feature, logits


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_iter = 5

    if args.dataID == 1:
        dataset_name = 'UCM'
        num_classes = 21
    elif args.dataID == 2:
        dataset_name = 'AID'
        num_classes = 30
    elif args.dataID == 3:
        dataset_name = 'NWPU_RESISC45'
        num_classes = 45
    else:
        raise ValueError(f"Unknown dataID: {args.dataID}")

    if args.surrogate_model == 'vit':
        base_encoder = 'vit_tiny_patch16_224.augreg_in21k_ft_in1k'
        batch_size = args.batch_size
        target_layer = 'blocks.0'
    elif args.surrogate_model == 'resnet18':
        base_encoder = 'resnet18'
        batch_size = args.batch_size
        target_layer = 'maxpool'
    elif args.surrogate_model == 'swin':
        base_encoder = 'swin_tiny_patch4_window7_224.ms_in1k'
        batch_size = args.batch_size
        target_layer = 'layers.0'
    elif args.surrogate_model == 'rsmamba':
        base_encoder = 'rsmamba_b'
        batch_size = args.batch_size
        target_layer = 'layers.0'
    elif args.surrogate_model == 'convnext':
        base_encoder = 'convnext_tiny.in12k_ft_in1k'
        batch_size = args.batch_size
        target_layer = 'stem'
    elif args.surrogate_model == 'replknet':
        base_encoder = 'replknet'
        batch_size = args.batch_size
        target_layer = 'stem.3'
    else:
        raise ValueError(f"Unknown surrogate_model: {args.surrogate_model}")

    data_path = f'./data/'

    chkpt_path = f'results/{dataset_name}/{args.surrogate_model}/chkpt/best.pth'
    # chkpt_path = None if not os.path.exists(chkpt_path) else chkpt_path

    if not os.path.exists(f'results/{dataset_name}/{args.surrogate_model}/chkpt/best.pth'):
        raise ValueError(f"Checkpoint not found: {chkpt_path}, please train the model first.")

    save_path_prefix = f'./{args.save_prefix}/{dataset_name}/{args.surrogate_model}_{target_layer}_iter{num_iter}/'
    print("Current Dataset：", dataset_name, "Adversarial Example Size:", args.crop_size, "Batch Size:", batch_size)

    surrogate_model = CLARS(base_encoder, pretrained_path=chkpt_path, target_layer=target_layer,
                            num_classes=num_classes).to(device)

    surrogate_model.eval()

    # follow MoCov3's augmentation recipe
    clean_augmentation = [
        transforms.Resize(size=(args.crop_size, args.crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]

    mix_augmentation = [
        transforms.Resize(size=(args.crop_size, args.crop_size)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.5),
        # transforms.RandomGrayscale(p=0.5),
        # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomApply([Solarize()], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]

    data_set = cls_dataset(root_dir=data_path, pathfile='./data/' + dataset_name + '_test.txt',
                           transform=TwoCropsTransform(transforms.Compose(clean_augmentation),
                                                       transforms.Compose(mix_augmentation)))

    dataloader = DataLoader(data_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    cls_loss = nn.CrossEntropyLoss()

    kl_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')

    alpha = args.alpha

    ... # to be continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataID', type=int, default=1, help='1: UCMerced_LandUse, 2: AID, 3: NWPU_RESISC45')
    parser.add_argument('--surrogate_model', type=str, default='resnet18',
                        help='vit, resnet18, swin, rsmamba, convnext, replknet')
    parser.add_argument('--save_prefix', type=str, default='adv_examples')
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epsilon', type=float, default=1)
    parser.add_argument('--decay', type=float, default=1)
    parser.add_argument('--alpha', type=float, default=1)

    strat_time = time.time()
    main(parser.parse_args())
    end_time = time.time()
    print('Time cost:', end_time - strat_time)
