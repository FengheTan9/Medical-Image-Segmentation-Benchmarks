import torch
import numpy as np
import os
import argparse
from torch.utils.data import DataLoader
from src.dataloader.dataset import MedicalDataSets
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
import src.utils.losses as losses
from src.utils.metrics import iou_score
from torchvision.utils import save_image
# Assuming model imports based on your provided training script
from src.network.conv_based.CMUNet import CMUNet
from src.network.conv_based.U_Net import U_Net
from src.network.conv_based.AttU_Net import AttU_Net
from src.network.conv_based.UNeXt import UNext
from src.network.conv_based.UNetplus import ResNet34UnetPlus
from src.network.conv_based.UNet3plus import UNet3plus
from src.network.conv_based.CMUNeXt import cmunext
from src.network.transfomer_based.transformer_based_network import get_transformer_based_model

def load_model(model_path, args, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    # Model selection based on argument
    if args.model == "CMUNet":
        model = CMUNet(output_ch=args.num_classes)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
            # model.cuda()
    elif args.model == "CMUNeXt":
        model = cmunext(num_classes=args.num_classes)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
            # model.cuda()
    elif args.model == "U_Net":
        model = U_Net(output_ch=args.num_classes)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
            # model.cuda()
    elif args.model == "AttU_Net":
        model = AttU_Net(output_ch=args.num_classes)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
            # model.cuda()
    elif args.model == "UNext":
        model = UNext(output_ch=args.num_classes)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
            # model.cuda()
    elif args.model == "UNetplus":
        model = ResNet34UnetPlus(num_class=args.num_classes)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
            # model.cuda()
    elif args.model == "UNet3plus":
        model = UNet3plus(n_classes=args.num_classes)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
            # model.cuda()
    else:
        # Adjust accordingly for transformer-based models
        model = get_transformer_based_model(model_name=args.model, img_size=args.img_size, num_classes=args.num_classes, in_ch=3)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def get_val_transform(img_size):
    return Compose([
        Resize(img_size, img_size),
        Normalize(),
        # ToTensorV2(),
    ])

def validate(model, val_loader, criterion, device, save_dir="validation_results"):
    """执行验证，并且每隔十张图像保存一次预测结果到PNG文件"""
    model.eval()
    val_loss = 0.0
    val_iou = 0.0
    val_dice = 0.0
    val_rvd = 0.0
    os.makedirs(save_dir, exist_ok=True)  

    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(val_loader):
            img_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            img_batch, label_batch = img_batch.to(device), label_batch.to(device)
            outputs = model(img_batch)
            loss = criterion(outputs, label_batch)

            val_loss += loss.item()

            iou, dice, rvd, _, _, _, _, _ = iou_score(outputs, label_batch)
            val_iou += iou
            val_dice += dice
            if rvd<1:
                val_rvd += rvd
            # 每隔十张图像保存一次预测结果
            if i_batch % 1 == 0:
                # 将模型输出转换为二值图像
                outputs = torch.sigmoid(outputs)
                outputs[outputs > 0.5] = 1
                outputs[outputs <= 0.5] = 0
                output_images = outputs.cpu().data
                
                # 保存图像
                for idx, img in enumerate(output_images):
                    save_path = os.path.join(save_dir, f"batch_{i_batch}_img_{idx}.png")
                    # 使用save_image从torchvision，或者使用其他方法将张量转换为图像并保存
                    save_image(img, save_path)

    val_loss /= len(val_loader)
    val_iou /= len(val_loader)
    val_dice /= len(val_loader)
    val_rvd /= len(val_loader)
    print(f'验证损失: {val_loss:.4f}, 验证IoU: {val_iou:.4f}, 验证dice：{val_dice:.4f}, 验证rvd：{val_rvd:.4f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validation script for medical image segmentation")
    parser.add_argument('--model', type=str, default="U_Net", help='model type')
    parser.add_argument('--model_path', type=str, default="./checkpoint/U_Net_model.pth", help='Path to the trained model')
    parser.add_argument('--base_dir', type=str, default="./data/test", help='base directory of dataset')
    parser.add_argument('--val_file_dir', type=str, default="test_val.txt", help='validation file directory')
    parser.add_argument('--img_size', type=int, default=256, help='image size')
    parser.add_argument('--num_classes', type=int, default=1, help='number of classes')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, args, device)

    val_transform = get_val_transform(args.img_size)

    db_val = MedicalDataSets(base_dir=args.base_dir, split="val", transform=val_transform, val_file_dir=args.val_file_dir)
    val_loader = DataLoader(db_val, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    criterion = losses.__dict__['BCEDiceLoss']().to(device)
    validate(model, val_loader, criterion, device)
