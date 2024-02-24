import numpy as np
import torch
import imbalanced_databases as imbd
import math
from MSM_tool.Kmeansmote2 import KmeanSmoteTrainingMarchine as Ks2
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
import sklearn.datasets as ds
from imblance_dataset.data_set import ImbalanceMultipleClassDataset as IMCD


def muticlass_generate_data(X, y, sampling_num, batch_size0):
    """
    uni, counts = np.unique(y, return_counts=True)

    normal_label = uni[np.argmax(counts)]
    for class_label in uni:
        if class_label != normal_label:
            generate_data(X, y, sampling_num, class_label)
    """
    uni, counts = np.unique(y, return_counts=True)
    msm_input, msm_output, msm_label = [], [], []
    for class_label, c in zip(uni, counts):
        input0, output0 = generate_data(X, y, sampling_num, class_label)
        msm_input.append(input0)
        msm_output.append(output0)
        one_hot_label = np.eye(len(uni))[class_label]
        msm_label.append(np.tile(one_hot_label, (input0.shape[0], 1)))

    msm_input = np.concatenate(msm_input, axis=0)
    msm_output = np.concatenate(msm_output, axis=0)
    msm_label = np.concatenate(msm_label, axis=0)

    msm_input = torch.from_numpy(msm_input).float()
    msm_output = torch.from_numpy(msm_output).float()
    msm_label = torch.from_numpy(msm_label).float()
    dataset = TensorDataset(msm_input, msm_output, msm_label)
    train_loader = DataLoader(dataset, batch_size=batch_size0, shuffle=True, drop_last=True)
    return train_loader


def generate_data(X, y, sampling_num, select_label):
    k = Ks2(X, y)
    x_samp, y_samp = k.kmeansmote_generate_data(select_label=select_label, sampling_num=sampling_num)  # 产生训练样本
    return x_samp, y_samp


class MutiMsmTrainModel(torch.nn.Module):
    def __init__(self, input_size, output_size, classNum, hidden_sizes=[64, 64, 32, 32]):
        super().__init__()

        layers = [torch.nn.Linear(input_size, hidden_sizes[0]), torch.nn.LeakyReLU()]
        for i in range(1, len(hidden_sizes)):
            layers.append(torch.nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            layers.append(torch.nn.LeakyReLU())

        self.condition_layer = torch.nn.Linear(hidden_sizes[-2] + classNum, hidden_sizes[-1])
        self.activation = torch.nn.LeakyReLU()
        self.out_layer = torch.nn.Linear(hidden_sizes[-1], output_size)

        self.network = torch.nn.ModuleList(layers)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, input, condition):

        x = input
        for layer in self.network:
            x = layer(x)
        x = torch.cat([x, condition], dim=1)
        x = self.activation(self.condition_layer(x))
        x = self.out_layer(x)
        return x


def coin_flip(input_x, label, iteration, model, conditon_class, k=25):
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
        concatenated_tensor = torch.cat((direction.to(device), input_x, scaling_factor.to(device)), dim=1)
        input_x = model.forward(concatenated_tensor, conditon_class)
        loss = criterion(input_x, label)

    return loss


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_default_tensor_type(torch.DoubleTensor)  # 代码中网络参数类型不统一

    # # 生成不平衡多分类数据集
    # X, y = make_classification(n_samples=300, n_features=2, n_informative=2, n_redundant=0, n_classes=3,
    #                            weights=[0.1, 0.4, 0.5], n_clusters_per_class=1, random_state=42)
    # data_name = 'demo'
    # 数据集
    imdataset = IMCD()

    my_data = imdataset.load_binary_iris_k_fold()
    my_data = my_data['train0']

    data_name = 'w'

    # my_data = imdataset.load_binary_wt()
    # data_name = 'eee'

    # 随机种子
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # 训练数据预处理
    STR = 'MutiMSM_' + data_name

    # 数据标准化
    X, y = my_data['data'], my_data['target']

    # 创建MinMaxScaler对象
    # scaler = MinMaxScaler()
    # X_Zscore = scaler.fit_transform(X)
    X_Zscore = X

    sampling_num = 1600
    bs = 128
    train_data = muticlass_generate_data(X_Zscore, y, sampling_num, bs)

    # 模型超参数
    epoch_num = 200
    input_size = 2 * len(X[0]) + 1
    output_size = len(X[0])  # 数据维数
    class_num = len(np.unique(y))

    # 损失函数
    criterion = torch.nn.MSELoss()

    # 模型
    msm_model = MutiMsmTrainModel(input_size, output_size, class_num).to(device)
    # 优化器
    optimizer = torch.optim.Adam(msm_model.parameters())

    # 训练
    for epoch in range(1, epoch_num + 1):
        loss_total = 0
        for batch in train_data:
            inputs, targets, yy = batch
            inputs, targets, yy = inputs.to(device), targets.to(device), yy.to(device)
            out = msm_model.forward(inputs.double(), yy)
            loss = coin_flip(out, targets.double(), epoch, msm_model, yy)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_total += loss.item()

        print('Epoch %d, Training loss %.2f' % (
            epoch, float(loss_total)))

        # 保存模型
        torch.save(msm_model.state_dict(), '../MutiMSM_train_parameters/' + STR + '.pth')
