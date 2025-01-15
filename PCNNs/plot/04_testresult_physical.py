import pandas as pd
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np

data = pd.DataFrame(pd.read_csv(r"D:\03_LSP\06_github\03_rf_geemapResult\03_rf_geemapResult\04_test_result_phesical.csv"))
print(data)

# print(bands)
# 创建柱状图
predicted= data["predicted"]
observed=data["observed"]

# 打印观测值和预测值列表
plt.scatter(observed, predicted, s=30)
# 绘制趋势线
z = np.polyfit(observed, predicted, 1)
p = np.poly1d(z)
plt.plot(observed, p(observed), color='red')

# plt.text(20, 30, rmseTraining, transform=plt.gca().transAxes)
rmse = np.sqrt(np.mean((predicted - observed) ** 2))
print("RMSE:", rmse)

# 在图表上显示 RMSE
plt.text(50,210,'R-squared='+str(r2_score(observed, predicted))[:4])
plt.text(50, 200, f"RMSE={rmse:.2f}")
# plt.text(50,180,'MAE='+str(mean_absolute_error(observed, predicted))[:5],fontdict={'size':16})
plt.text(50,190,'MAE='+str(mean_absolute_error(observed, predicted))[:5])
# 在图表上显示 RMSE


# 设置图表标题和轴标签
plt.title('Predicted vs Observed')
plt.xlabel('observed')
plt.ylabel('predicted')


plt.savefig(r'D:\03_LSP\06_github\03_rf_geemapResult\01_result\04_SOS_test_physical.png', dpi=300)
plt.show()