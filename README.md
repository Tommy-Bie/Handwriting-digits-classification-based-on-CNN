# Handwriting-digits-classification-based-on-CNN
use CNN based on AlexNet/VGG/GoogLeNet to implement the classification of handwriting digits (not MNIST) -  pytorch

使用卷积神经网络（基于AlexNet/VGG-16/GoogLeNet）实现了对手写数字数据集的识别分类

数据集为非MNIST手写数字集（由模式识别课程老师提供）共0-9十个数字，每个数字55张图片，存放在digit文件夹中，已整理规整，建议使用torchvision.datasets.ImageFolder读取数据集

图像预处理：中心裁剪+压缩

损失函数：交叉熵

优化方法：带动量的随机梯度下降

结果（测试正确率）：

AlexNet：95.45%

VGG16：92.73%

**GoogLeNet：97.27%**
