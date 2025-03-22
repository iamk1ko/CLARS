import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import timm
from datasets.cls_dataset import cls_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from engine import *
from models.replknet import create_RepLKNet31B
from models.rsmamba import RSMamba
import warnings

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
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

    print("Current Dataset：", dataset_name, "Data Path", args.data_path, "Victim Model:", args.victim_model)

    data_path = args.data_path

    chkpt_path = f'results/{dataset_name}/{args.victim_model}/chkpt/best.pth'
    if not os.path.exists(chkpt_path):
        raise ValueError(f"Checkpoint not found: {chkpt_path}")

    if args.victim_model == 'resnet18':
        model = timm.create_model("resnet18", pretrained=False, num_classes=num_classes)
    elif args.victim_model == 'vit':
        model = timm.create_model('vit_tiny_patch16_224.augreg_in21k_ft_in1k', pretrained=False,
                                  num_classes=num_classes)
    elif args.victim_model == 'swin':
        model = timm.create_model('swin_tiny_patch4_window7_224.ms_in1k', pretrained=False,
                                  num_classes=num_classes)
    elif args.victim_model == 'rsmamba':
        model = RSMamba(arch='b', out_type='avg_featmap', img_size=args.crop_size,
                        num_classes=num_classes, patch_cfg=dict(stride=8),
                        init_cfg=[dict(type='Kaiming', layer='Conv2d', mode='fan_in', nonlinearity='linear')])
    elif args.victim_model == 'convnext':
        model = timm.create_model('convnext_tiny.in12k_ft_in1k', pretrained=False, num_classes=num_classes)
    elif args.victim_model == 'replknet':
        model = create_RepLKNet31B(num_classes=num_classes, small_kernel_merged=False, use_checkpoint=False)
    else:
        raise ValueError(f"Unknown victim_model: {args.victim_model}, but you can add it in the code")

    clean_augmentation = [
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]

    data_set = cls_dataset(root_dir=data_path, pathfile='./data/' + args.datasets + '_test.txt',
                           transform=transforms.Compose(clean_augmentation))
    data_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = model.to(device)
    model.load_state_dict(torch.load(chkpt_path, map_location=device))
    print('====== resnet ======:')
    pred_one_epoch_classification(data_loader, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataID', type=int, default=1, help='1: UCMerced_LandUse, 2: AID, 3: NWPU_RESISC45')
    parser.add_argument('--victim_model', type=str, default='resnet18',
                        help='vit, resnet18, swin, rsmamba, convnext, replknet')
    parser.add_argument('--data_path', type=str, default='adv_examples')
    parser.add_argument('--batch_size', type=int, default=8)

    strat_time = time.time()
    main(parser.parse_args())
    end_time = time.time()
    print('Time cost:', end_time - strat_time)
