
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


# # 模型预测
# model.eval()

# 使用训练好的模型进行预测
with torch.no_grad():
    # y_pred = model(X_test)
    X_test = torch.from_numpy(xt_test).float().to(device)
    print(type(X_test))
    # y_pred = calcError(PINN(X_test))
    result, y_test_inputs, y_pred = calcError(PINN(X_test))

print(y_test, y_pred)
print(y_test.shape, len(y_pred))

# 将预测结果和真实结果还原为原始数据
# y_pred = scaler.inverse_transform(y_pred.values.reshape(-1, 1)).flatten()
# y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
# 输出结果
# results = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred})
# print(results)

print("Test: ")
print('R_square:', r2_score(y_test, y_pred))
print('RMSE:', mean_squared_error(y_test, y_pred, squared=False))
print('MAE:', mean_absolute_error(y_test, y_pred))