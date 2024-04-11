# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.io as sio
from lt_Unet_GIDC import DGI_reconstruction, image_cut_by_std
from PIL import Image
import numpy as np


image_name = 'CelebA'  # name of your image used for simulation
dataSet = 'CelebA_128_GRAY'  # 'CelebA' or 'stl10'
dim = 64
num_patterns = 1024
NN = 'Unet'
mode = 'wDGI'


# learned patterns
# 加载训练好的模式和随机模式
pattern_save_path = '.\\model\\trained_%s_patterns_%d_%s_%s_%d.mat'%(dataSet,num_patterns,NN,mode,dim)
trained_patterns = sio.loadmat(pattern_save_path)['trained_patterns']
trained_patterns = torch.from_numpy(trained_patterns).float()


# random patterns
random_patterns = torch.randn((1, num_patterns, dim, dim), dtype=torch.float32)
random_patterns.clamp_(min=0, max=1)
random_patterns = (random_patterns > 0).float()
random_patterns = torch.randn((num_patterns, dim, dim), dtype=torch.float32)
random_patterns = random_patterns.view(1, num_patterns, dim, dim)
random_patterns.clamp_(min=0, max=1)
random_patterns = (random_patterns > 0).float()




# 加载图像并进行预处理
image_path = '.\\data\\images\\%s.bmp' % image_name
with Image.open(image_path) as img:
    im = img.convert('L')  # 转换为灰度图
im = im.resize((dim, dim), Image.BICUBIC)  # 使用双三次插值
im_np = np.array(im)
im_np = (im_np - im_np.min())/ (im_np.max() - im_np.min())
im_np = im_np.reshape(1, 1, dim, dim)
im_tensor = torch.from_numpy(im_np)
# 如果需要，可以指定数据类型为float32
im_tensor = im_tensor.float()
im_tensor = im_tensor.expand(-1, num_patterns, -1, -1)  # 在通道维度上扩展为 num_patterns 个通道


# 逐元素相乘
train_multiplied = trained_patterns * im_tensor
y_learn = train_multiplied.sum(dim=(2, 3)).view(1, num_patterns, 1, 1)
random_multiplied = random_patterns * im_tensor
y_rand = random_multiplied.sum(dim=(2, 3)).view(1, num_patterns, 1, 1)

# 计算均值和标准差，并进行标准化
mean_y_learn, var_y_learn = torch.mean(y_learn, dim=(0, 1, 2, 3), keepdim=True), torch.std(y_learn, dim=(0, 1, 2, 3), keepdim=True)
y_learn = (y_learn - mean_y_learn) / (var_y_learn + 1e-8).sqrt()

mean_y_rand, var_y_rand = torch.mean(y_rand, dim=(0, 1, 2, 3), keepdim=True), torch.std(y_rand, dim=(0, 1, 2, 3), keepdim=True)
y_rand = (y_rand - mean_y_rand) / (var_y_rand + 1e-8).sqrt()

# DGI重建（需要DGI_reconstruction函数的实现）
dgi_learn = DGI_reconstruction(y_learn, trained_patterns, num_patterns, dim, dim, 0.5)
dgi_rand = DGI_reconstruction(y_rand, random_patterns, num_patterns, dim, dim, 0.5)

# 将张量转换为numpy数组并保存
y_learn_s = y_learn.detach().numpy().reshape([num_patterns, 1])
dgi_learn_s = dgi_learn.detach().numpy().reshape([dim, dim])
y_rand_s = y_rand.detach().numpy().reshape([num_patterns, 1])
dgi_rand_s = dgi_rand.detach().numpy().reshape([dim, dim])

# 保存结果
im_np = np.array(im)
sio.savemat('.\\data\\' + image_name + '_sim.mat', {'y': y_learn_s, 'dgi_r': dgi_learn_s, 'GT': im_np})

# 后处理技巧以更好地可视化（需要image_cut_by_std函数的实现）
dgi_learn_post = image_cut_by_std(dgi_learn_s, 2)
dgi_rand_post = image_cut_by_std(dgi_rand_s, 2)

random_patterns = np.reshape(random_patterns[:, 1, :, :], [dim, dim])
trained_patterns = np.reshape(trained_patterns[:, 1, :, :], [dim, dim])

plt.subplot(231)
plt.imshow(random_patterns)
plt.title('rand pattern')
plt.axis('off')
plt.subplot(232)
plt.plot(y_rand_s)
plt.title('measurments')
plt.axis('off')
plt.subplot(233)
plt.imshow(dgi_rand_post)
plt.title('DGI-rand')
plt.axis('off')

plt.subplot(234)
plt.imshow(trained_patterns)
plt.axis('off')
plt.title('learned pattern')
plt.subplot(235)
plt.plot(y_learn_s)
plt.title('measurments')
plt.axis('off')
plt.subplot(236)
plt.imshow(dgi_learn_post)
plt.title('DGI-learn')
plt.axis('off')

plt.show()


























