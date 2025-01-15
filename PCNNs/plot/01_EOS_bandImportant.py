import pandas as pd
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


data = pd.DataFrame(pd.read_csv(r"D:\03_LSP\06_github\03_rf_geemapResult\03_rf_geemapResult\05_EOS_bandimportance.csv"))
print(data)

# print(bands)
# 创建柱状图
bands = data["Variable"]
importance=data["Importance"]

plt.figure(figsize=(20,12))


plt.bar(bands, importance)

# 设置图表标题和轴标签
plt.title('Random Forest Variable Importance')
plt.xlabel('Bands')
plt.ylabel('Importance')
# 自动调整图表布局
# plt.tight_layout()
# 显示 x 轴每个值
plt.xticks(bands, bands)  # 设置 x 轴刻度标签和旋转角度为 90 度



plt.savefig(r'D:\03_LSP\06_github\03_rf_geemapResult\01_EOS_rf_valueImportant.png', dpi=300)

plt.show()


