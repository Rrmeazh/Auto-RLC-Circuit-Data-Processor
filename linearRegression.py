import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, data_path):
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
        self.k = np.zeros(1) # 拟合斜率
        self.b = np.zeros(1) # 拟合截距
        self.r = np.zeros(1) # 相关系数

    def fit(self):
        '''
        计算拟合斜率、截距和相关系数
        '''
        X = np.vstack((np.ones(len(self.x)), self.x)).T
        Y = self.lny
        B = np.linalg.solve(X.T @ X, X.T @ Y)
        Y_pred = X @ B

        self.b = B[0]
        self.k = B[1]
        self.r = np.corrcoef(self.lny, Y_pred)[0, 1]

    def plot(self, image_path):
        '''
        绘制散点图和拟合直线
        '''
        plt.scatter(self.x, self.lny, label='Data Points')
        x_fit = np.linspace(min(self.x), max(self.x), 100)
        y_fit = self.k * x_fit + self.b
        plt.plot(x_fit, y_fit, color='red', label='Fitted Line')
        plt.xlabel(self.x_name)
        plt.ylabel(self.y_name)
        plt.title(f'Linear Regression (r={self.r:.4f})')
        plt.legend()
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        plt.savefig(image_path)
        plt.show()

if __name__ == "__main__":
    script_path = os.path.dirname(os.path.abspath(__file__))
    lr = LinearRegression(os.path.join(script_path, 'data/data1.csv'))
    lr.fit()
    print(f'Fitted line: y = {lr.k:.4f}x + {lr.b:.4f}')
    print(f'Correlation coefficient: r = {lr.r:.4f}')
    lr.plot(os.path.join(script_path, 'result/images/linear_regression.png'))