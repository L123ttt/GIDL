# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io as sio
from torch.utils.data import DataLoader
import pickle
import matplotlib.pyplot as plt


def train_model(epoch, num_train_imgs, batch_size, dataSet, num_patterns, NN, mode, dim, train_dataset_path, val_dataset_path, net, lr, img_W, img_H, show_val_loss, show_data):

    # 设置是否使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    step_num = int(epoch * num_train_imgs / batch_size + 1)
    epoch_step = int(num_train_imgs / batch_size)

    model_save_path = f"./model/model_{dataSet}_{num_patterns}_{NN}_{mode}_{dim}.pt"
    pattern_save_path = f"./model/trained_{dataSet}_patterns_{num_patterns}_{NN}_{mode}_{dim}.mat"
    loss_save_path = f"./model/loss_{dataSet}_{NN}_{mode}_{num_patterns}_{dim}.mat"

    # 加载数据集文件
    with open(train_dataset_path, 'rb') as f:
        train_dataset = pickle.load(f)

    with open(val_dataset_path, 'rb') as f:
        val_dataset = pickle.load(f)
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    # 从数据加载器中获取批量的图像和标签数据
    train_images_batch, train_labels_batch = next(iter(train_loader))
    val_images_batch, val_labels_batch = next(iter(val_loader))
    # 将数据移动到GPU上
    train_images_batch, train_labels_batch = train_images_batch.to(device), train_labels_batch.to(device)
    val_images_batch, val_labels_batch = val_images_batch.to(device), val_labels_batch.to(device)

    net = net.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.99), eps=1e-8)

    count = 0
    train_loss = []
    running_loss = 0.0
    for step in range(step_num):

        optimizer.zero_grad()
        DGI_r, patterns, x_out = net(train_images_batch, batch_size, num_patterns, img_W, img_H)
        loss = criterion(x_out, train_labels_batch)
        print("loss on train batch : ", loss.item())
        loss.backward()
        optimizer.step()

        # 模型验证及结果绘制
        if show_val_loss:
            if step % epoch_step == 0:
                with torch.no_grad():
                    # 获取验证数据的模型输出
                    val_DGI_r, val_patterns, val_out = net(val_images_batch, batch_size, num_patterns, img_W, img_H)
                    # 计算验证损失
                    val_loss = criterion(val_out, val_labels_batch)
                    # 打印验证损失
                    print("******************************************************************************************")
                    print('[all_step_num : {}] - [epoch : {}] - [step for epoch : {}]'.format(step_num, epoch, epoch_step))
                    print('[step {}]: loss on validation batch: {:.8f}'.format(step, val_loss.item()))
                    print("******************************************************************************************")
                    if show_data:
                        # 绘制模型预测结果图像
                        x_pred = val_out.cpu().numpy()  # 将张量移动回CPU并转换为NumPy数组
                        x_pred = np.reshape(x_pred[count, :, :, :], [img_W,img_H])
    
                        learned_pattern = val_patterns.cpu().numpy()
                        learned_pattern = np.reshape(learned_pattern[:, count, :, :], [img_W, img_H])
    
                        DGI_pred = val_DGI_r.cpu().numpy()
                        DGI_pred = np.reshape(DGI_pred[count, :, :, :], [img_W,img_H])
    
                        ground_truth = val_labels_batch.cpu().numpy()
                        ground_truth = np.reshape(ground_truth[count, :, :, :], [img_W, img_H])
    
                        count = count + 1
    
                        plt.figure(figsize=(12, 4))
                        plt.subplot(141)
                        plt.imshow(learned_pattern)
                        plt.title('Learned Pattern')
                        plt.axis('off')
    
                        plt.subplot(142)
                        plt.imshow(DGI_pred)
                        plt.title('DGI Pred.')
                        plt.axis('off')
    
                        plt.subplot(143)
                        plt.imshow(x_pred)
                        plt.title('DL Pred.')
                        plt.axis('off')
    
                        plt.subplot(144)
                        plt.imshow(ground_truth)
                        plt.title('Ground Truth')
                        plt.axis('off')
                        plt.subplots_adjust(wspace=0.3)
                        plt.show()
        running_loss += loss.item()
        train_loss.append(running_loss / step_num)
    # Save model
    torch.save(net.state_dict(), model_save_path)
    # Save trained patterns
    trained_patterns = patterns.detach().cpu().numpy()  # 将张量移动回CPU
    sio.savemat(pattern_save_path, {'trained_patterns': trained_patterns})
    # Save loss
    sio.savemat(loss_save_path, {'train_loss': train_loss})
