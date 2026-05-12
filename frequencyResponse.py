import numpy as np
import pandas as pd
import os

class FrequencyResponse:
    def __init__(self, data_path: str) -> None:
        '''
        读入csv文件的数据以初始化对象属性
        默认以逗号分隔，第一行为表头，第一列为频率，第二列为V_R峰-峰值，第三列为相位差
        '''
        data = pd.read_csv(data_path, delimiter=',')
        self.freq_name = data.columns[0]
        self.vpp_name = data.columns[1]
        self.phase_name = data.columns[2]
        self.freq = data.values[:, 0]
        self.vpp = data.values[:, 1]
        self.phase = data.values[:, 2]
        self.f0 = self.freq[np.argmax(self.vpp)] # 共振频率对应的频率值
        self.vpp_max = np.max(self.vpp) # 峰-峰值的最大值
        self.f1 = None # 3dB带宽的下限频率
        self.f2 = None # 3dB带宽的上限频率


    def find_3db_points(self) -> None:
        '''
        找到3dB带宽的频率上下限
        '''
        vpp_3db = np.max(self.vpp) / np.sqrt(2)
        vpp_minus_3db = self.vpp - vpp_3db
        indices = np.where(np.diff(np.sign(vpp_minus_3db)))[0]
        if len(indices) != 2:
            raise ValueError("3dB点数量不为2，请检查数据是否正确")

        # 线性插值计算3dB点的频率
        self.f1 = self.freq[indices[0]] + (vpp_3db - self.vpp[indices[0]]) * (self.freq[indices[0] + 1] - self.freq[indices[0]]) / (self.vpp[indices[0] + 1] - self.vpp[indices[0]])
        self.f2 = self.freq[indices[1]] + (vpp_3db - self.vpp[indices[1]]) * (self.freq[indices[1] + 1] - self.freq[indices[1]]) / (self.vpp[indices[1] + 1] - self.vpp[indices[1]])

    def report(self) -> tuple[float, float, float, float, float, float]:
        '''
        输出共振频率、峰-峰值最大值、3dB带宽与品质系数
        Returns:
            self.f0: 共振频率
            self.vpp_max: 峰-峰值最大值
            self.f1: 3dB带宽的下限频率
            self.f2: 3dB带宽的上限频率
            Delta_f: 3dB带宽
            Q: 品质系数
        '''
        self.find_3db_points()
        bandwidth = self.f2 - self.f1
        Q = self.f0 / bandwidth
        return self.f0, self.vpp_max, self.f1, self.f2, bandwidth, Q
    
if __name__ == "__main__":
    script_path = os.path.dirname(os.path.abspath(__file__))
    fr = FrequencyResponse(os.path.join(script_path, 'data', 'data2.csv'))
    f0, vpp_max, f1, f2, bandwidth, Q = fr.report()
    print(f'共振频率: {f0:.2f} Hz')
    print(f'峰-峰值最大值: {vpp_max:.2f} V')
    print(f'3dB带宽下限频率: {f1:.2f} Hz')
    print(f'3dB带宽上限频率: {f2:.2f} Hz')
    print(f'3dB带宽: {bandwidth:.2f} Hz')
    print(f'品质系数: Q = {Q:.2f}')