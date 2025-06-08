import os
import warnings
import imgaug
from tqdm import tqdm
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from monai.optimizers import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
import numpy as np
from monai.utils import MetricReduction
import torch
import config
from torch import nn
from nets.basicUnet import UNetTaskAligWeight1
from nets.basicUnet_new import UNetTaskAligWeight
from util import logger, metrics, common,loss
from torch.utils.data import DataLoader,random_split
import torch.nn.functional as F
from monai.metrics import DiceMetric, HausdorffDistanceMetric,MeanIoU
from torchmetrics import Recall, Precision, ConfusionMatrix, F1Score, Accuracy, AUROC
from torch.utils.tensorboard import SummaryWriter  # 导入SummaryWriter
from torch.optim.lr_scheduler import LambdaLR,ReduceLROnPlateau
from random import  uniform,random
from monai.utils import set_determinism
import random
from util.data_utils import CDDataAugmentation,CDDataAugmentation1
from util.loss import DC_and_BCE_loss
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from medpy.metric import hd95
from util.roi import process_and_augment_roi
import pywt

def wavelet_enhance_rgb_for_unet(img_rgb):
    # Step 1: 转换为 YCrCb
    img_ycc = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(img_ycc)

    # Step 2: 对亮度 Y 通道做小波增强
    coeffs = pywt.dwt2(y, 'haar')  # 或 'db2' / 'sym4'
    cA, (cH, cV, cD) = coeffs

    # Step 3: 增强高频部分
    cH *= 1.5
    cV *= 1.5
    cD *= 1.5

    # Step 4: 重构增强后的 Y 通道
    enhanced_y = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
    enhanced_y = np.clip(enhanced_y, 0, 255).astype(np.uint8)

    # Step 5: 合并并转换回 RGB
    enhanced_img_ycc = cv2.merge((enhanced_y, cr, cb))
    enhanced_rgb = cv2.cvtColor(enhanced_img_ycc, cv2.COLOR_YCrCb2RGB)

    return enhanced_rgb

class CustomDataset(Dataset):
    def __init__(self, model,image_list, path, img_size=224, crop=None, is_train=None):
        self.image_list = image_list
        self.image_folder = path
        self.img_size = img_size
        self.model=model
        self.augm1=CDDataAugmentation1(img_size=224,
                                           ori_size=224, crop=crop,p_hflip=0.0,p_vflip=0.0, color_jitter_params=None,
                                           long_mask=True)
        # 加载分类标签表
        label_path = r"/data/coding/pythonProject/BUSI/data/train/train_classification_label.xlsx"
        df = pd.read_excel(label_path)
        self.class_dict = dict(zip(df['Image'].astype(str).str.zfill(4), df['Pterygium']))

        if is_train:
            self.augm = CDDataAugmentation(img_size=img_size,
                                           ori_size=img_size, crop=True, p_hflip=0.7, p_vflip=0.7, p_rota=0.5,
                                           p_scale=0.6,
                                           p_gaussn=0.5, p_contr=0.6, p_gama=0.5, p_distor=0.6,
                                           color_jitter_params=None,
                                           p_random_affine=0,
                                           long_mask=True)
        else:
            self.augm = CDDataAugmentation(img_size=img_size,
                                           ori_size=img_size, crop=crop, p_hflip=0.0, p_vflip=0.0,
                                           color_jitter_params=None,
                                           long_mask=True)


    def __len__(self):
        return len(self.image_list)

    def correct_dims(self, *images):
        corr_images = []
        for img in images:
            if len(img.shape) == 2:
                corr_images.append(np.expand_dims(img, axis=2))
            else:
                corr_images.append(img)
        if len(corr_images) == 1:
            return corr_images[0]
        else:
            return corr_images

    def __getitem__(self, idx):
        sub_id = self.image_list[idx]  # 如 '0001'
        img_path = os.path.join(self.image_folder, sub_id, f"{sub_id}.png")
        label_path = os.path.join(self.image_folder, sub_id, f"{sub_id}_label.png")
        image = cv2.imread(img_path, 1)
        # image = wavelet_enhance_rgb_for_unet(image)
        se_label = cv2.imread(label_path, 0)
        se_label[se_label == 38] = 1
        cl_label = int(self.class_dict[sub_id])
        image, se_label = self.correct_dims(image, se_label)
        image=self.augm1.transform(image)
        device = torch.device("cuda" )
        image = process_and_augment_roi(self.model,image,device,self.augm1)
        image, se_label = self.augm.transform(image, se_label)
        se_label = se_label.unsqueeze(0)
        return {
            "image": image,
            "cl_label": cl_label,
            "se_label": se_label,
        }


def val(model,val_loader, loss_se,loss_cl, device):
    model.eval()
    running_loss = 0.0
    seg_total_loss = 0.0
    cl_total_loss = 0.0
    #三个常用的医学图像分割评估指标
    dice_metric = DiceMetric(include_background=False,reduction=MetricReduction.MEAN)
    hausdorff_metric = HausdorffDistanceMetric(include_background=False, distance_metric="euclidean",percentile=95,reduction=MetricReduction.MEAN)
    iou_metric = MeanIoU(include_background=False)
    f1_metric = F1Score(num_classes=3, average='macro', task='multiclass').to(device)
    acc_metric = Accuracy(num_classes=3, average='macro', task='multiclass').to(device)
    auc_metric = AUROC(num_classes=3, average='macro', task='multiclass').to(device)
    matrix2 = ConfusionMatrix(num_classes=3, task='multiclass').to(device)
    with torch.no_grad():
        for idx, data in enumerate(val_loader):
            imgs = data['image'].float().to(device)
            se_label = data['se_label'].float().to(device)
            cl_label = data['cl_label'].long().to(device)
            se_out,  cl_out = model(imgs)
            # cl_out = torch.squeeze(cl_out,dim=1)
            se_loss = loss_se(se_out, se_label).to(device)
            criterion = torch.nn.CrossEntropyLoss()
            cl_loss = criterion(cl_out, cl_label)
            seg_total_loss += se_loss
            cl_total_loss += cl_loss
            loss = 0.7 * se_loss + 0.3 * cl_loss
            running_loss += loss.item()
            se_out = F.sigmoid(se_out)
            pred_masks = torch.where(se_out > 0.5, torch.tensor(1.0).to(device), torch.tensor(0.0).to(device))
            pred_masks_np = pred_masks.cpu().numpy()
            se_label_np = se_label.cpu().numpy()
            # 计算 HD95
            hd95_value = hd95(pred_masks_np, se_label_np)
            # 计算Dice指数
            for i in range(pred_masks.size(0)):
                if torch.all(pred_masks[i] == 0):
                    pred_masks[i][0][0][0] = 1
            dice_metric(pred_masks, se_label)
            # 计算Hausdorff距离
            hausdorff_metric(pred_masks, se_label)
            iou_metric(pred_masks, se_label)
            predict_cl = torch.argmax(F.softmax(cl_out, dim=-1), dim=-1).to(device)
            auc_metric(cl_out, cl_label)
            f1_metric(predict_cl, cl_label)
            acc_metric(predict_cl, cl_label)
            matrix2(predict_cl, cl_label)
    print("seg - {:.4f},cl - {:.4f}".format( seg_total_loss / len(val_loader), cl_total_loss / len(val_loader)))
    dice_score = dice_metric.aggregate().item()
    # hausdorff_distance = hausdorff_metric.aggregate().item()
    iou_score = iou_metric.aggregate().item()
    f1_score = f1_metric.compute().cpu().numpy()
    acc_score = acc_metric.compute().cpu().numpy()
    auc_score = auc_metric.compute().cpu().numpy()
    matrix2_score = matrix2.compute().cpu().numpy()
    torch.cuda.empty_cache()
    return running_loss / len(val_loader),dice_score,hd95_value,iou_score,f1_score,acc_score,auc_score,matrix2_score,seg_total_loss/len(val_loader)


def train(model,optimizer, train_loader,loss_se,loss_cl,device,epoch,n=2):
    print("=======Epoch:{}=======lr:{}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
    model.train()
    running_loss = 0.0
    seg_total_loss = 0.0
    cl_total_loss = 0.0
    tempPred = 0
    for idx, data in enumerate(train_loader):
        imgs = data['image'].float().to(device)
        se_label = data['se_label'].float().to(device)
        cl_label = data['cl_label'].long().to(device)
        for i in range(n):
            if i == 0:
                se_out, cl_out = model(imgs)
                tempPred = se_out.detach()
            else:
                tempPred = F.sigmoid(tempPred).to(device)
                # diff = custom_function(tempPred,device)
                # pred_masks = torch.where(tempPred > 0.5, torch.tensor(1.0).to(device), torch.tensor(0.0).to(device))
                # diff = se_label -pred_masks
                batch_size, num_channels, height, width = tempPred.size()
                # feature_map_flat = tempPred.view(batch_size, num_channels, -1)  # 平均池化，得到 (batch_size, num_channels)
                # weight = model.weight(feature_map_flat).view(batch_size, 1, 1,1)  # 计算权重，并调整为 (batch_size, 1, 1, 1)
                # projected_images = imgs+tempPred*weight
                diff = (torch.abs(0.5-tempPred)*2).view(batch_size, -1)
                diff = torch.sum(diff,dim=1)/(diff.size()[1])
                diff = diff.view(batch_size, 1, 1,1)
                imgs=imgs +tempPred * diff
                se_out, cl_out = model(imgs)
            se_loss = loss_se(se_out, se_label).to(device)
            criterion = torch.nn.CrossEntropyLoss()
            cl_loss = criterion(cl_out, cl_label)
            #
            loss = 0.6 * se_loss + 0.4 * cl_loss

            # loss = model.loss_function.cuda()(se_loss, cl_loss)
            optimizer.zero_grad()
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
            seg_total_loss += se_loss
            cl_total_loss += cl_loss
            running_loss += loss.item()
    print(" seg - {:.4f},cl - {:.4f}".format( seg_total_loss / len(train_loader)/n,cl_total_loss / len(train_loader)/n))
    torch.cuda.empty_cache()
    return running_loss / len(train_loader)/n,seg_total_loss / len(train_loader)




def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    set_determinism(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




# 定义交叉验证函数
# 定义交叉验证函数
def cross_validation(args):
    train_folder = '/data/coding/pythonProject/BUSI/data/train'
    val_folder = train_folder

    # 读取标签表格
    label_df = pd.read_excel(r"/data/coding/pythonProject/BUSI/data/train/train_classification_label.xlsx")
    label_df['Image'] = label_df['Image'].astype(str).str.zfill(4)  # 保证图像编号像 '0001'

    # 遍历每个图像，补全缺失的 label 图
    for sub_id in tqdm(label_df['Image']):
        folder_path = os.path.join(train_folder, sub_id)
        img_path = os.path.join(folder_path, f"{sub_id}.png")
        label_path = os.path.join(folder_path, f"{sub_id}_label.png")

        if os.path.exists(img_path) and not os.path.exists(label_path):
            # 自动生成空白分割标签图
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            blank_label = np.zeros((h, w), dtype=np.uint8)
            cv2.imwrite(label_path, blank_label)

    # 最终使用标签表格作为图像列表来源（不会漏掉标签为0的图）uda
    all_folders = label_df['Image'].tolist()

    # 划分训练集和验证集（最后100张为验证）
    val_image_list = all_folders[-50:]
    train_image_list = all_folders[:-50]
    device = torch.device('cuda')
    import shutil
    save_path = os.path.join("checkpoint", "Zhou")

    # 删除目录及其内容
    if os.path.exists(save_path):
        try:
            shutil.rmtree(save_path)  # 删除目录及其中的所有内容
        except Exception as e:
            print(f"删除目录失败: {e}")
            # 这里可以进行其他错误处理

    # 重新创建目录
    os.makedirs(save_path)

    # 初始化 SummaryWriter
    #writer = SummaryWriter(log_dir=save_path)

    lr_threshold = 0.0001
    seed_everything(args.seed)

    model1 = UNetTaskAligWeight1(n_channels=3, n_classes=1).to(device)
    checkpoint = torch.load("/data/coding/pythonProject/trained_model/best_model_epoch173_se_only.pt", map_location=device)
    model1.load_state_dict(checkpoint['net'])
    model1.to(device)
    train_dataset = CustomDataset(model1,train_image_list, train_folder, img_size=args.img_size, is_train=True)
    val_dataset = CustomDataset(model1,val_image_list, val_folder, img_size=args.img_size, is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    model = UNetTaskAligWeight(n_channels=3,n_classes=1).to(device)
    checkpoint = torch.load("/data/coding/pythonProject/trained_model/best_model_epoch153.pt", map_location=device)
    model.load_state_dict(checkpoint['net'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, )
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, threshold=0.001,
                                  threshold_mode="abs", min_lr=0.00001)  # gamma是衰减因子

    se_loss = DC_and_BCE_loss(dice_weight=0.5).to(device)
    cl_loss = loss.BCEFocalLoss(gamma=2,alpha=0.4).to(device)

    start_epoch = 0

    best_val_index = 0
    best_min_loss = 2
    early_stop_patience = 50  # 设置早停的阈值
    early_stop_counter = 0
    best_model_path = ''
    best_seg_model_path =' '
    model_path=''
    for epoch in range(start_epoch,args.epochs):

        train_loss,train_seg_loss = train(model,optimizer, train_loader,se_loss,cl_loss,device, epoch)
        val_loss, dice_score,hd_score,iou_score,f1_score,acc_score,auc_score,matrix2_score,val_seg_loss = val(model,val_loader,se_loss,cl_loss,device)

        scheduler.step(train_loss)

        print(f"Epoch_[{epoch}/{args.epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Epoch [{epoch}/{args.epochs}] - Dice Score: {dice_score:.4f}  - Hd Score: {hd_score:.4f} - iou_score: {iou_score:.4f}"
              f" f1: {f1_score:.4f}, acc: {acc_score:.4f} ,auc: {auc_score:.4f},,matrix_score2:{matrix2_score}",flush=True)
        index =  dice_score


        if val_loss < best_min_loss:
            best_min_loss = val_loss
            early_stop_counter = 0
            try:
                os.remove(best_model_path)
            except OSError as e:
                print(OSError)
            best_model_path = f"{save_path}/best_model_epoch{epoch}.pt"
            state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, best_model_path)
            os.chmod(best_model_path, 0o777)
        else:
            early_stop_counter += 1

        print(f"early_stop_counter_[{early_stop_counter}]")
        if early_stop_counter > early_stop_patience:
            if optimizer.param_groups[0]['lr'] >= lr_threshold:
                print("My patience ended, but I believe I need more time")
                early_stop_counter = early_stop_counter - 20
            else:
                print("Early stoping epoch!!", epoch)
                break
        if index > best_val_index:
            best_val_index = index
            try:
                os.remove(best_seg_model_path)
            except OSError as e:
                print(OSError)
            best_seg_model_path = f"{save_path}/best_seg_model_epoch{epoch}.pt"
            state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, best_seg_model_path)
            os.chmod(best_seg_model_path, 0o777)
        if epoch %20==0:
            model_path = f"{save_path}/model_epoch{epoch}.pt"
            state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, model_path)
            os.chmod(model_path, 0o777)
    del model
    # 关闭TensorBoard记录器
   # writer.close()

if __name__ == '__main__':
    args = config.args
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)


    # 进行交叉验证
    cross_validation(args)
