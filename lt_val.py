# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.io as sio
from lt_Unet_GIDC import create_model
from PIL import Image
import numpy as np
from lt_Unet_GIDC import DGI_reconstruction


def finetune(image_path, dataSet, num_patterns, NN, mode, dim, img_W, img_H, finetune_steps, pre_model_path, finetune_lr, finetune_freq):
    # 设置是否使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载训练好的patterns
    pattern_save_path = '.\\model\\trained_%s_patterns_%d_%s_%s_%d.mat' % (dataSet, num_patterns, NN, mode, dim)
    trained_patterns = sio.loadmat(pattern_save_path)['trained_patterns']
    trained_patterns = torch.from_numpy(trained_patterns).float().to(device)

    # 加载已保存的模型参数
    model = create_model(img_W, img_H, num_patterns)
    model.load_state_dict(torch.load(pre_model_path), strict=False)
    model.to(device)
    # 设置为评估模式
    model.eval()

    # 默认设置所有参数为不可训练
    for param in model.parameters():
        param.requires_grad = False

    # # 仅设置前三层的参数为可训练
    # for param in model.conv1.parameters():
    #     param.requires_grad = True
    # for param in model.conv1_1.parameters():
    #     param.requires_grad = True
    # for param in model.conv2.parameters():
    #     param.requires_grad = True
    # for param in model.conv2_1.parameters():
    #     param.requires_grad = True
    # for param in model.conv3.parameters():
    #     param.requires_grad = True
    # for param in model.conv3_1.parameters():
    #     param.requires_grad = True

    # 仅设置后三层的参数为可训练
    for param in model.conv8.parameters():
        param.requires_grad = True
    for param in model.conv8_1.parameters():
        param.requires_grad = True
    for param in model.conv8_2.parameters():
        param.requires_grad = True
    for param in model.conv9.parameters():
        param.requires_grad = True
    for param in model.conv9_1.parameters():
        param.requires_grad = True
    for param in model.conv9_2.parameters():
        param.requires_grad = True
    for param in model.conv10.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=finetune_lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    loss_function = nn.MSELoss()  # 或者使用 'sum' 或 'none'
    # 设置为训练模式
    model.train()

    # 加载图像并进行预处理
    with Image.open(image_path) as img:
        im = img.convert('L')  # 转换为灰度图
    im = im.resize((dim, dim), Image.BICUBIC)
    im_np = np.array(im)
    im_np = im_np / 255.0
    im_np = im_np.reshape(1, 1, dim, dim)
    im_tensor0 = torch.from_numpy(im_np).to(device)
    # 如果需要，可以指定数据类型为float32
    im_tensor = im_tensor0.float()
    # 在通道维度上扩展为 num_patterns 个通道
    im_tensor = im_tensor.expand(-1, num_patterns, -1, -1)
    # 逐元素相乘
    train_multiplied = trained_patterns * im_tensor
    y_learn = train_multiplied.sum(dim=(2, 3)).view(1, num_patterns, 1, 1)
    # 计算均值和标准差，并进行标准化
    mean_y_learn, var_y_learn = torch.mean(y_learn, dim=(0, 1, 2, 3), keepdim=True), torch.std(y_learn, dim=(0, 1, 2, 3), keepdim=True)
    y_learn = (y_learn - mean_y_learn) / (var_y_learn + 1e-8).sqrt()
    # DGI重建（需要DGI_reconstruction函数的实现）
    dgi_learn = DGI_reconstruction(y_learn, trained_patterns, num_patterns, dim, dim, 0.5).to(device)

    # 微调预训练模型
    for step in range(finetune_steps):
        optimizer.zero_grad()
        DGI_r, _, x_out = model(dgi_learn, 1, num_patterns,trained_patterns, img_W, img_H)
        # 计算损失
        loss = loss_function(x_out, im_tensor)
        print("step : ", step, "  ---loss on train batch : ", loss.item())
        loss.backward()
        optimizer.step()

        if step % finetune_freq == 0:
            # 绘制模型预测结果图像
            x_pred = x_out.detach().cpu().numpy()  # 将张量移动回CPU并转换为NumPy数组
            x_pred = np.reshape(x_pred[0, :, :, :], [img_W, img_H])
            im_tensor1 = im_tensor0.cpu().numpy()  # 将张量移动回CPU并转换为NumPy数组
            im_tensor1 = np.reshape(im_tensor1[0, 0, :, :], [img_W, img_H])

            plt.subplot(221)
            plt.imshow(x_pred)
            plt.title('finetune')
            plt.axis('off')

            plt.subplot(222)
            plt.imshow(im_tensor1)
            plt.title('original')
            plt.axis('off')

            plt.subplot(223)
            plt.imshow(x_pred,cmap="gray")
            plt.title('finetune-gray')
            plt.axis('off')

            plt.subplot(224)
            plt.imshow(im_tensor1,cmap="gray")
            plt.title('original-gray')
            plt.axis('off')

            plt.show()
