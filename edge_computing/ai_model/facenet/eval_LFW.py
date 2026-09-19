import torch
import torch.backends.cudnn as cudnn

from nets.facenet import Facenet
from utils.dataloader import LFWDataset
from utils.utils_metrics import test

if __name__ == "__main__":
    #--------------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #--------------------------------------#
    cuda            = True
    #--------------------------------------#
    #   主干特征提取网络的选择
    #   mobilenet
    #   inception_resnetv1
    #--------------------------------------#
    backbone        = "mobilenet"
    # backbone        ='inception_resnetv1'

    #--------------------------------------------------------#
    #   输入图像大小，常用设置如[112, 112, 3]
    #--------------------------------------------------------#ep1000-loss0.000-val_loss0.669.pth
    input_shape     = [256, 256, 3]
    #--------------------------------------#
    #   训练好的权值文件
    #--------------------------------------#facenet_mobilenet.pth
    model_path = "savemodel_HFB_mix/k7andk3_withedloss_epoch6000/ep6000-loss0.000-val_loss1.174.pth"


    # model_path      ='savemodel_csist_lab2_unet256_k8andk8_withoutedloss_train_withmobilenet/ep100-loss0.003-val_loss0.001.pth'
    # model_path ='model_data/facenet_mobilenet.pth'
    #--------------------------------------#
    #   LFW评估数据集的文件路径
    #   以及对应的txt文件
    #--------------------------------------#
    lfw_dir_path    = "datasets//VSDN//X4//k7andk3_withedloss"
    lfw_pairs_path  = "pairs/vsdn/x4/k7andk3_withedloss.txt"


    #--------------------------------------#
    #   评估的批次大小和记录间隔
    #--------------------------------------#
    batch_size      = 256
    log_interval    = 1
    #--------------------------------------#
    #   ROC图的保存路径fakefake
    #--------------------------------------#
    png_save_path   = "figure/x4_k7k7withoutedloss.png"

    test_loader = torch.utils.data.DataLoader(
        LFWDataset(dir=lfw_dir_path, pairs_path=lfw_pairs_path, image_size=input_shape), batch_size=batch_size, shuffle=False)


    model = Facenet(backbone=backbone, mode="predict")

    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model  = model.eval()

    if cuda:
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model = model.cuda()



    test(test_loader, model, png_save_path, log_interval, batch_size, cuda)
