import numpy as np
import torch
import imbalanced_databases as imbd
import math
from MSM_tool.Kmeansmote2 import KmeanSmoteTrainingMarchine as Ks2
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from imblance_dataset.data_set import ImbalanceMultipleClassDataset as IMCD

def generate_data(X, y, sampling_num, select_label):
    k = Ks2(X, y)
    x_samp, y_samp = k.kmeansmote_generate_data(select_label=select_label, sampling_num=sampling_num)  # 产生训练样本

    x_tensor = torch.from_numpy(x_samp).float()
    y_tensor = torch.from_numpy(y_samp).float()
    dataset = TensorDataset(x_tensor, y_tensor)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

    return train_loader


class MsmTrainModel(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[128, 64, 32]):
        super().__init__()
        layers = [torch.nn.Linear(input_size, hidden_sizes[0]), torch.nn.LeakyReLU()]
        for i in range(1, len(hidden_sizes)):
            layers.append(torch.nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            layers.append(torch.nn.LeakyReLU())
        layers.append(torch.nn.Linear(hidden_sizes[-1], output_size))

        self.network = torch.nn.ModuleList(layers)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, input):
        x = input
        for layer in self.network:
            x = layer(x)
        return x


def coin_flip(input_x, label, iteration, model, k=25):
    epsilon = k / (k + np.exp(iteration / k))
    if np.random.rand(1) <= epsilon:
        loss = criterion(input_x, label)
    else:
        b, d = input_x.shape[0], input_x.shape[1]
        random_tensor = 2 * torch.rand(b, d) - 1
        # 计算每行向量的范数
        norms = random_tensor.norm(dim=1, keepdim=True)
        # 归一化每行的向量
        direction = random_tensor / norms
        scaling_factor = torch.rand(b, 1)
        # 使用 torch.cat 将它们拼接在一起
        concatenated_tensor = torch.cat((direction, input_x, scaling_factor), dim=1)
        input_x = model.forward(concatenated_tensor)
        loss = criterion(input_x, label)

    return loss


if __name__ == '__main__':
    torch.set_default_tensor_type(torch.DoubleTensor)  # 代码中网络参数类型不统一
    imdataset = IMCD()
    # 数据集
    my_data = imdataset.load_binary_wt()
    data_name = 'binary_wt'
    #
    # X, y = my_data['data'], my_data['target']

    # my_data = imdataset.load_binary_wt()
    # my_data = my_data['train4']
    # data_name = 'wt_k5'

    # 随机种子
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # 数据标准化
    X, y = my_data['data'], my_data['target']
    # 训练数据预处理
    STR = 'MSM_' + data_name
    label_all, counts = np.unique(y, return_counts=True)
    minority_idx = int(label_all[np.argmin(counts)])
    # 数据标准化
    scaler = StandardScaler()
    X_Zscore = scaler.fit_transform(X)
    # X_Zscore = X
    sampling_num = 3000
    train_data = generate_data(X_Zscore, y, sampling_num, minority_idx)

    # 模型超参数
    epoch_num = 150
    input_size = 2 * len(X[0]) + 1
    output_size = len(X[0])  # 数据维数

    # 损失函数
    criterion = torch.nn.MSELoss()

    # 模型
    msm_model = MsmTrainModel(input_size, output_size)
    # 优化器
    optimizer = torch.optim.Adam(msm_model.parameters())

    # 训练
    for epoch in range(1, epoch_num + 1):
        loss_total = 0
        for batch in train_data:
            inputs, targets = batch
            out = msm_model.forward(inputs.double())
            loss = coin_flip(out, targets.double(), epoch, msm_model)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_total += loss.item()

        print('Epoch %d, Training loss %.2f' % (
            epoch, float(loss_total)))

        # 保存模型
        torch.save(msm_model.state_dict(), '../MSM_train_parameters/' + STR + '.pth')

