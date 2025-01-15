import sys
from sklearn.model_selection import train_test_split
sys.path.append(".")
import numpy as np
import torch
from torch.autograd import grad
from network import DNN
from scipy.io import loadmat
import pandas as pd
import torch.nn as nn
import numpy as np
import os

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(1234)
np.random.seed(1234)

data = pd.DataFrame(pd.read_csv(r"D:\02_github\01_DBF_sites\SOSdaymet568.csv"))
u = data['SOS']
x = data.iloc[:,-46:]
# print(x.columns)
# 特征选择
# x = data.iloc[:, [2, 3, 4, 5, 6,7, -46, -45, -44, -43, -42, -41, -40, -39, -38, -37, -36, -35, -34, -33, -32, -31, -30, -29, -28, -27, -26, -25, -24, -23, -22, -21, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1]]
t = data.iloc[:, [2, 3, 4, 5, 6,7]]
# print(t.columns)

dayl = data.iloc[:,2]
# print(dayl)
prcp = data.iloc[:,3]
srad = data.iloc[:,4]
tmax = data.iloc[:,5]
tmin = data.iloc[:,6]
vp = data.iloc[:,7]


print(x.shape,t.shape)
print(u.shape)

ub = np.array([np.array(x).max(), np.array(t).max()])
lb = np.array([np.array(x).min(), np.array(t).min()])
print(ub,lb)



x_, t_ = x, t
# x_, t_ = np.asarray(x),np.asarray(t)
# print(x_.shape, t_.shape)

x_ = x_.values.reshape(-1, 1)
t_ = t_.values.reshape(-1, 1)
u_ = u.values.reshape(-1, 1)
print(x_.shape, t_.shape,u_.shape)


N_u = 200
rand_idx = np.random.choice(len(u), N_u, replace=False) #选择索引这行代码从 u_ 的索引中随机选择 N_u 个索引，并将结果存储在 rand_idx 中。这可能是为了从数据中选择部分样本用于训练或其他用途
# 所以这部分可以变为分为0.8 和 0.2的比例。
x = torch.tensor(x_[rand_idx], dtype=torch.float32).to(device)
t = torch.tensor(t_[rand_idx], dtype=torch.float32).to(device)
xt = torch.cat((x, t), dim=1)
print("xt",xt.shape)
u = torch.tensor(u_[rand_idx], dtype=torch.float32).to(device)
print(x.shape,t.shape)
print(u.shape,xt.shape)

# print(u)
# print(xt)




def calcError(pcnn):
    u_pred = pcnn.net(torch.hstack((x, t)))
    u_pred = u_pred.detach().cpu().numpy()
    u_ = u.detach().cpu().numpy()

    print("u_",u_)
    print("u_pred",u_pred)


    error_u = np.linalg.norm(u_ - u_pred, 2) / np.linalg.norm(u_, 2)
    lambda1 = pcnn.lambda_1.detach().cpu().item()
    lambda2 = np.exp(pcnn.lambda_2.detach().cpu().item())
    error_lambda1 = np.abs(lambda1 - 1.0) * 100
    error_lambda2 = np.abs(lambda2 - 0.01 / np.pi) * 100
    print(
        f"\nError u  : {error_u:.5e}",
        f"\nError l1 : {error_lambda1:.5f}%",
        f"\nError l2 : {error_lambda2:.5f}%",
    )
    return (error_u, error_lambda1, error_lambda2)

if __name__ == "__main__":
    pcnn = PCNNs(u)
    pcnn.optimizer.step(pcnn.closure)
    torch.save(pcnn.net.state_dict(), r"D:\02_github\03_PCNNs\weight_clean3.pt")
    pcnn.net.load_state_dict(torch.load(r"D:\02_github\03_PCNNs\weight_clean3.pt"))
    calcError(pcnn)

