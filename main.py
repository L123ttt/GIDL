# -*- coding: utf-8 -*-

import torch
from lt_data import datatset_pre                # 准备数据集
from lt_Unet_GIDL import create_model           # 搭建模型
from lt_train import train_model                # 训练模型
from lt_val import finetune                     # 微调模型


def main():
    print("- -- --- > 准备数据集 ：训练集+验证集")
    datatset_pre(img_W, img_H, image_folder, train_dataset_path, val_dataset_path)

    print("- -- --- > 创建模型 ：模型和结构+权重参数")
    net = create_model(img_W, img_H, num_patterns)

    print("- -- --- > 开始训练模型（U-net），保存patterns、预训练模型")
    train_model(epoch, num_train_img, batch_size, dataSet, num_patterns, NN, mode, dim, train_dataset_path, val_dataset_path, net, train_lr, img_W, img_H, True, False)

    print("- -- --- > 调用patterns，微调预训练模型")
    finetune(image_path, dataSet, num_patterns, NN, mode, dim, img_W, img_H, finetune_steps, pre_model_path, finetune_lr, finetune_freq)


if __name__ == "__main__":
    """
        在此定义整个工程代码的可调参数，无需在实际代码中调整参数
            1、数据准备
                训练集、验证集
            2、模型训练 
                ①、模型搭建
                学习率
                优化器
                loss
                训练过程展示
                ②、保存结果
                    pattern
                    U-net
            3、调用patterns，微调预训练模型
                更新后三层模型参数
                
    """

    """     可修改参数     """
    dim = 64                                    # 用于实验的图片尺寸大小
    # 模型超参数
    epoch = 64
    train_lr = 0.0002
    finetune_lr = 0.0002
    batch_size = 256

    num_patterns = 1024                         # 采样率 = num_patterns / (img_W * img_H)
    num_train_img = 29000                       # 训练集的图片数量
    finetune_freq = 100                         # finetune过程打印效果的间隔步数
    finetune_steps = 1000                       # finetune过程最大步数
    image_path = './data/images/CelebA.bmp'     # finetune过程使用的图片

    """     不建议修改参数     """
    img_W = dim
    img_H = dim
    lab_W = dim
    lab_H = dim
    NN = 'Unet'
    mode = 'wDGI'
    data_name = 'CelebA_sim'
    dataSet = 'CelebA'                                                                              # 训练集文件夹名
    image_folder = './data/%s' % dataSet                                                            # 存放图片的路径
    val_dataset_path = './dataset/val_dataset.pkl'                                                  # 验证集路径
    train_dataset_path = './dataset/train_dataset.pkl'                                              # 训练集路径
    pre_model_path = "./model/model_%s_%d_%s_%s_%d.pt" % (dataSet, num_patterns, NN, mode, dim)     # 预训练模型
    torch.manual_seed(1)

    """     进入工程的主函数     """
    main()
