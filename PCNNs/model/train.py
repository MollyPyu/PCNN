import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.autograd import grad
from pyDOE import lhs


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(1234)
np.random.seed(1234)
# 读取数据
data = pd.DataFrame(pd.read_csv(r"D:\03_LSP\06_github\11_PhenoCam_SOS_EOS\SOSdaymet568.csv"))


y = data['SOS']
xt = data.iloc[:,[2, 3, 4, 5, 6, 7, -46, -45, -44, -43, -42, -41, -40, -39, -38, -37, -36, -35, -34, -33, -32, -31, -30, -29, -28,-27, -26, -25, -24, -23, -22, -21, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4,-3, -2, -1]]
# x = data.iloc[:,-46:] #选择后365个数值
# t = data.iloc[:, [2, 3, 4, 5, 6, 7]]

dayl = data.iloc[:, 2]
prcp = data.iloc[:, 3]
srad = data.iloc[:, 4]
tmax = data.iloc[:, 5]
tmin = data.iloc[:, 6]
vp = data.iloc[:, 7]

dayl = torch.tensor(dayl,dtype=torch.float32).to(device)
prcp = torch.tensor(prcp,dtype=torch.float32).to(device)
srad  = torch.tensor(srad ,dtype=torch.float32).to(device)
tmax = torch.tensor(tmax,dtype=torch.float32).to(device)
tmin = torch.tensor(tmin,dtype=torch.float32).to(device)
vp = torch.tensor(vp,dtype=torch.float32).to(device)
# 数据归一化
scaler = StandardScaler()
xt = scaler.fit_transform(xt)
y = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

ub = np.array([np.array(xt).max()])
lb = np.array([np.array(xt).min()])
ub = torch.tensor(ub,dtype=torch.float).to(device)
lb = torch.tensor(lb,dtype=torch.float).to(device)

# lambda_val = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])  # Weight parameters of the constraints
lambda_val = torch.tensor([0.22, 0.16, 0.14, 0.16, 0.15, 0.17], dtype=torch.float)

xt =  torch.tensor(xt,dtype=torch.float32).to(device)
y =  torch.tensor(y,dtype=torch.float32).to(device)

N_u = 100
N_f = 10000
x_min = -1
x_max = 1
t_min = -1
t_max = 1

xt_f = x_min + (x_max - x_min) * lhs(1, N_f)
xt_f = torch.tensor(xt_f, dtype=torch.float32).to(device)


# 划分训练集和测试集
xt_train, xt_test, y_train, y_test = train_test_split(xt, y, test_size=0.2, random_state=2)
print(xt_test.shape,y_test.shape)



def calcError(pinn):
    u_pred = pinn.net(xt)
    u_pred = u_pred.detach().cpu().numpy()
    u_ = y.detach().cpu().numpy()
    error_u = np.linalg.norm(u_ - u_pred, 2) / np.linalg.norm(u_, 2)

    y_train_pred = u_pred[:, -1]
    y_train_pred = scaler.inverse_transform(y_train_pred.reshape(-1, 1)).flatten()
    y_train = scaler.inverse_transform(u_.reshape(-1, 1)).flatten()

    return (error_u,y_train, y_train_pred)

# 创建模型实例
input_size = xt.shape[1]
model = PINN(input_size)

# 定义损失函数和优化器
criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# optimizer = optim.Adam(model.f(xt))
# 定义优化器
optimizer = model.optimizer
# 训练模型
num_epochs = 10

for epoch in range(num_epochs):
    # model.train()
    train_loss = 0.0
    model.iter = epoch  # 更新迭代次数，用于输出信息
    model.optimizer.zero_grad()
    # for inputs, labels in train_loader:
    for u in xt_train:
        pinn = PINN(u)
        # print(pinn)
        # pinn = PINN(inputs)  #<__main__.PINN object at 0x000002242CAFCBD0>
        pinn.optimizer.step(pinn.closure)
        result, y_train, y_train_pred = calcError(pinn)
        # print(result)
        loss = model.closure()
    train_loss /= len(xt_train)
    # 打印训练信息
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}')


# ###############train restult#####################
# y_train_pred = [:, -1].detach().numpy()
# y_train_pred = scaler.inverse_transform(y_train_pred.reshape(-1, 1)).flatten()
# y_train = scaler.inverse_transform(labels.reshape(-1, 1)).flatten()

print("Train: ")
print("RMSE=", mean_squared_error(y_train, y_train_pred))
print('R_square for RNN:', r2_score(y_train, y_train_pred))
print('MAE=', mean_absolute_error(y_train, y_train_pred))
print('mean squared error', mean_squared_error(y_train, y_train_pred) ** 0.5)
print(y_train, y_train_pred)

