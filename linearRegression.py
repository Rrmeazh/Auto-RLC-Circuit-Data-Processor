import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import scipy.stats as stats

class LinearRegression:
    def __init__(self, data_path: str) -> None:
        '''
        读入csv文件的数据
        默认以逗号分隔，第一行为表头，第一列为x，第二列为y
        '''
        data = pd.read_csv(data_path, delimiter=',')
        self.x_name = data.columns[0]
        self.y_name = 'ln(' + data.columns[1] + ')' # 视情况取合适变换
        self.x = data.values[:, 0]
        self.y = data.values[:, 1]
        self.lny = np.log(self.y) # 视情况取合适变换
        self.lny_pred = np.zeros_like(self.lny) # 拟合预测值
        self.k = np.zeros(1) # 拟合斜率
        self.b = np.zeros(1) # 拟合截距
        self.r = np.zeros(1) # 相关系数
        self.confidence_rate = 0.95 # 置信水平
        self.confidence_band = (np.zeros_like(self.lny), np.zeros_like(self.lny)) # 置信区间带

    def fit(self) -> None:
        '''
        计算拟合斜率、截距和相关系数
        '''
        X = np.vstack((np.ones(len(self.x)), self.x)).T
        Y = self.lny
        B = np.linalg.solve(X.T @ X, X.T @ Y)
        self.lny_pred = X @ B
        self.b = B[0]
        self.k = B[1]
        self.r = np.corrcoef(self.lny, self.lny_pred)[0, 1]
    
    def calc_confidence_band(self) -> None:
        '''
        计算整个回归线的置信区间带
        '''
        n = len(self.x)
        s_squared = np.sum((self.lny - self.lny_pred) ** 2) / (n - 2) # 均方误差
        F = stats.f.ppf(self.confidence_rate, 2, n - 2) # F分布临界值
        s_mu_x = np.sqrt(s_squared * (1/n + (self.x - np.mean(self.x))**2 / np.sum((self.x - np.mean(self.x))**2))) # 预测值的标准差
        self.confidence_band = (self.lny_pred - np.sqrt(2 * F) * s_mu_x, self.lny_pred + np.sqrt(2 * F) * s_mu_x)

    def plot(self, image_path: str) -> None:
        '''
        绘制散点图、拟合直线与置信区间带
        '''
        plt.scatter(self.x, self.lny, label='Data Points')
        x_fit = np.linspace(min(self.x), max(self.x), 100)
        y_fit = self.k * x_fit + self.b
        plt.plot(x_fit, y_fit, color='red', label='Fitted Line')
        if self.confidence_band[0].any() and self.confidence_band[1].any():
            plt.fill_between(self.x, self.confidence_band[0], self.confidence_band[1], color='pink', alpha=0.5, label='Confidence Band')
        plt.xlabel(self.x_name)
        plt.ylabel(self.y_name)
        plt.title(f'Linear Regression (r={self.r:.4f}, confidence rate={self.confidence_rate:.2%})')
        plt.legend()
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        plt.savefig(image_path)
        plt.show()

if __name__ == "__main__":
    script_path = os.path.dirname(os.path.abspath(__file__))
    lr = LinearRegression(os.path.join(script_path, 'data', 'data1_temp.csv'))
    lr.fit()
    lr.calc_confidence_band()
    print(f'Fitted line: y = {lr.k:.4f}x + {lr.b:.4f}')
    print(f'Correlation coefficient: r = {lr.r:.4f}')
    lr.plot(os.path.join(script_path, 'result', 'images', 'linear_regression.png'))