import os
import shutil
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as sps
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

sns.set_style("darkgrid")
if os.path.isdir("normal_distribution/result2"):
    shutil.rmtree("normal_distribution/result2")
os.makedirs(f"normal_distribution/result2/")

file = pd.read_csv("data.csv")
file_limits = pd.read_csv("limits.csv")
dict_columns = {}
for i, val in enumerate(file.columns.tolist()):
    if i == 0:
        continue
    data = val.split("_")
    if dict_columns.get(data[0]) is None:
        dict_columns[data[0]] = [val]
    else:
        dict_columns.get(data[0]).append(val)
print(dict_columns)

for u in dict_columns:
    for dist in np.linspace(0.1, 0.5, 5):
        train, test = train_test_split(file, test_size=dist, random_state=42)
        print(u, dist)
        columns = dict_columns[u]
        time_labels = [int(i.split("_")[1]) for i in columns]
        time_labels_np = np.array(time_labels).reshape(-1, 1)
        time_check = max(time_labels)
        limits = file_limits[file_limits["name"] == u]
        bottom_limit = float(limits["bottom_limit"])
        top_limit = float(limits["top_limit"])
        f = open(f"normal_distribution/result2/{u}.txt", "a", encoding="utf-8")

        mean = np.array([np.array(train[i]).mean() for i in columns])
        std = np.array([np.array(train[i]).std() for i in columns])

        model_mean = LinearRegression()
        model_mean.fit(time_labels_np, mean)
        mean_coef = model_mean.coef_[0]
        predict_values = model_mean.predict(time_labels_np)

        model_std = LinearRegression()
        model_std.fit(time_labels_np, std)
        std_coef = model_std.coef_[0]
        predict_values = model_std.predict(time_labels_np)

        mean_test_time0 = test[columns[0]].mean()
        std_test_time0 = test[columns[0]].std()
        mean_test = []
        std_test = []
        for i in time_labels:
            mean_test.append(mean_test_time0 + mean_coef * i)
            std_test.append(std_test_time0 + std_coef * i)

        test_working_predict = []
        for i, j in zip(mean_test, std_test):
                test_working_predict.append(
                    sps.norm(loc=i, scale=j).cdf(top_limit) - sps.norm(loc=i, scale=j).cdf(bottom_limit))

        test_working_experiment = []
        for i in columns:
            working = 0
            for j in test[i]:
                if j > bottom_limit and j < top_limit:
                    working += 1
            test_working_experiment.append(working / test.shape[0])

        test_error = 0
        for p, e in zip(test_working_predict, test_working_experiment):
            test_error += pow((p - e) / e, 2)
        test_error = np.sqrt(test_error / len(test_working_predict))

        # test_error = abs((test_working_predict[6] - test_working_experiment[6]) / test_working_experiment[6])

        f.write(f"Обучающая выборка - {100 - int(dist * 100)}%, тестовая выборка - {int(dist * 100)}%\n")
        for t, p, e in zip(time_labels, test_working_predict, test_working_experiment):
            f.write(f" {t}: {p} ({e})\n")
        f.write(f" Ошибка: {test_error}\n\n")

        f.close()
