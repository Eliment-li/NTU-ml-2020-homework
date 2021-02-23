# 由前 9 個小時的 18 個 features (包含 PM2.5)預測的 10 個小時的 PM2.5。文档
import numpy as np
import pandas as pd


# Load rain.csv
def LoadData(path):
    data = pd.read_csv(path, encoding='big5')
    data = data.iloc[:, 3:]
    data[data == 'NR'] = 0
    raw_data = data.to_numpy()
    return raw_data

# Preprocessing


# Extract Features


# Training

# Testing


# Prediction

if __name__ == '__main__':
    raw_data = LoadData('d:/train.csv')
    month_data = {}
    for month in range(12):
        sample = np.empty([18, 480])
        for day in range(20):
            sample[:, day * 24: (day + 1) * 24] = raw_data[18 * (20 * month + day): 18 * (20 * month + day + 1), :]
        month_data[month] = sample
    x = np.empty([12 * 471, 18 * 9], dtype=float)
    y = np.empty([12 * 471, 1], dtype=float)

    for month in range(12):

        for day in range(20):

            for hour in range(24):

                if day == 19 and hour > 14:
                    continue
                x[month * 471 + day * 24 + hour, :] = month_data[month][:,
                                                      day * 24 + hour: day * 24 + hour + 9].reshape(1,
                                                                                                    -1)  # vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
                y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9]  # value
    print(x)
    # print(y)
