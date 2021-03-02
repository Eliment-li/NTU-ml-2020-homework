#
import numpy as np
import pandas as pd

# Homework 1: Linear Regression
# 由前 9 個小時的 18 個 features (包含 PM2.5)預測的 10 個小時的 PM2.5。文档
# 参考 https://colab.research.google.com/drive/131sSqmrmWXfjFZ3jWSELl8cm0Ox5ah3C#scrollTo=p9FfatPz6MU3

if __name__ == '__main__':
    # Load rain.csv
    data = pd.read_csv('d:/train.csv', encoding='big5')

    # Preprocessing
    data = data.iloc[:, 3:]
    data[data == 'NR'] = 0

    raw_data = data.to_numpy()

    # Extract Features (1)
    month_data = {}

    for month in range(12):
        sample = np.empty([18, 480])
        for day in range(20):
            sample[:, day * 24: (day + 1) * 24] = raw_data[18 * (20 * month + day): 18 * (20 * month + day + 1), :]
        month_data[month] = sample

    # Extract Features (2)
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
    print(y)
    # Normalize (1)
    mean_x = np.mean(x, axis=0)  # 18 * 9
    std_x = np.std(x, axis=0)  # 18 * 9
    for i in range(len(x)):  # 12 * 471
        for j in range(len(x[0])):  # 18 * 9
            if std_x[j] != 0:
                x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]

    dim = 18 * 9 + 1
    w = np.zeros([dim, 1])
    x = np.concatenate((np.ones([12 * 471, 1]), x), axis=1).astype(float)
    learning_rate = 100
    iter_time = 1000000
    adagrad = np.zeros([dim, 1])
    eps = 0.0000000001
    for t in range(iter_time):
        loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2)) / 471 / 12)  # rmse
        if (t % 100 == 0):
            print(str(t) + ":" + str(loss))
        gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y)  # dim*1
        adagrad += gradient ** 2
        w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
    np.save('weight.npy', w)
