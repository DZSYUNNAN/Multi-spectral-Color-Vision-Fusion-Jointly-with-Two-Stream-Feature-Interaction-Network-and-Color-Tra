import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
from models import fusion_model
from tqdm import tqdm
from PIL import Image
from torchvision.datasets import ImageFolder
from input_data import ImageDataset
from pytorch_ssim import ssim
import time
from torchvision.utils import save_image

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)

parser = argparse.ArgumentParser()
parser.add_argument("--infrared_dataroot", default="F:\dzs\shujuji\INO\meimiao10zhen/INO_TreesAndRunner_IR/", type=str)
parser.add_argument("--visible_dataroot", default="F:\dzs\shujuji\INO\meimiao10zhen/INO_TreesAndRunner_VIS/", type=str)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--output_root", default="./outputs/kernel3_10_share_net2/INO_TreesAndRunner_RGB/", type=str)
parser.add_argument("--image_size", type=int, default=[320, 448])
parser.add_argument("--epoch", type=int, default=1)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/")

if __name__ == "__main__":
    opt = parser.parse_args()
    if not os.path.exists(opt.checkpoint_dir):
        os.makedirs(opt.checkpoint_dir)
    device = torch.device('cuda:0')
    if not os.path.exists(opt.output_root):
        os.makedirs(opt.output_root)
    net = fusion_model.Fusion_share_Net2().cuda()
    net.load_state_dict(torch.load("./checkpoints/fusion_last_CONV_ssim_std_tv_vis_share2_10.pth"))
    net.eval()
    transform = transforms.Compose([
        #transforms.Resize(opt.image_size, interpolation=2),
        transforms.ToTensor()])
    n = 558
    with torch.no_grad():
        for i in range(n):
            start = time.time()
            index = i + 1
            infrared = Image.open(opt.infrared_dataroot + str(index) + '.jpg')
            infrared = transform(infrared).unsqueeze(0)
            visible = Image.open(opt.visible_dataroot + str(index) + '.jpg')
            visible = transform(visible).unsqueeze(0)
            infrared = infrared.cuda()
            visible = visible.cuda()
            fused_img1= net(infrared[:,0:1,:,:] , visible[:,0:1,:,:])
            fused_img2 = net(infrared[:, 1:2, :, :], visible[:, 1:2, :, :])
            fused_img3 = net(infrared[:, 2:3, :, :], visible[:, 2:3, :, :])
            fused_img = torch.cat([torch.cat([fused_img1,fused_img2],dim=1),fused_img3],dim=1)
            save_image(fused_img.cpu(), os.path.join(opt.output_root, str(index) + ".jpg"))
            end = time.time()
            print('consume time:',end-start)