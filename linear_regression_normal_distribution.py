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
if os.path.isdir("result"):
    shutil.rmtree("result")

file = pd.read_csv("data.csv")
file_limits = pd.read_csv("limits.csv")
train, test = train_test_split(file, test_size=0.2, random_state=42)
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
    print(u)
    columns = dict_columns[u]
    time_labels = [int(i.split("_")[1]) for i in columns]
    time_check = max(time_labels)
    limits = file_limits[file_limits["name"] == u]
    bottom_limit = float(limits["bottom_limit"])
    top_limit = float(limits["top_limit"])
    os.makedirs(f"result/{u}")
    f = open(f"result/{u}/results.txt", "w", encoding="utf-8")

    plt.figure(figsize=(11, 7))
    for i in range(file.shape[0]):
        data = np.array(file[columns].iloc[i])
        plt.title('')
        plt.xlabel('time')
        plt.ylabel(u)
        plt.plot(time_labels, data, color='steelblue', linewidth=1)
    plt.axhline(y=bottom_limit, xmin=0, xmax=1, color='k', linewidth=1.5)
    plt.axhline(y=top_limit, xmin=0, xmax=1, color='k', linewidth=1.5)
    plt.savefig(f"result/{u}/1.jpg")
    plt.close()

    plt.figure(figsize=(11, 7))
    for i in range(train.shape[0]):
        data = np.array(train[columns].iloc[i])
        plt.title('Train data')
        plt.xlabel('time')
        plt.ylabel(u)
        plt.plot(time_labels, data, color='steelblue', linewidth=1)
    plt.axhline(y=bottom_limit, xmin=0, xmax=1, color='k', linewidth=1.5)
    plt.axhline(y=top_limit, xmin=0, xmax=1, color='k', linewidth=1.5)
    plt.savefig(f"result/{u}/1_1.jpg")
    plt.close()

    plt.figure(figsize=(11, 7))
    for i in range(test.shape[0]):
        data = np.array(test[columns].iloc[i])
        plt.title('Test data')
        plt.xlabel('time')
        plt.ylabel(u)
        plt.plot(time_labels, data, color='steelblue', linewidth=1)
    plt.axhline(y=bottom_limit, xmin=0, xmax=1, color='k', linewidth=1.5)
    plt.axhline(y=top_limit, xmin=0, xmax=1, color='k', linewidth=1.5)
    plt.savefig(f"result/{u}/1_2.jpg")
    plt.close()

    plt.figure(figsize=(11, 7))
    for i in range(file.shape[0]):
        data = np.array(file[columns].iloc[i])
        plt.title('')
        plt.xlabel('time')
        plt.ylabel(u)
        plt.scatter(time_labels, data, color='steelblue', marker=".")
    plt.axhline(y=bottom_limit, xmin=0, xmax=1, color='k', linewidth=1.5)
    plt.axhline(y=top_limit, xmin=0, xmax=1, color='k', linewidth=1.5)
    plt.savefig(f"result/{u}/2.jpg")
    plt.close()

    mean = []
    std = []
    for i in columns:
        data = np.array(train[i])
        mean.append(data.mean())
        std.append(data.std())

    f.write("Мат. ожидания:\n")
    f.write(str(mean))
    f.write("\nСреднеквадратические отклонения:\n")
    f.write(str(std))

    mean = np.array(mean)
    std = np.array(std)
    time_labels_np = np.array(time_labels).reshape(-1, 1)

    model_mean = LinearRegression()
    model_mean.fit(time_labels_np, mean)
    mean_coef = model_mean.coef_[0]
    predict_values = model_mean.predict(time_labels_np)
    plt.figure(figsize=(11, 7))
    plt.title('Linear Regression (mean)')
    plt.xlabel('time')
    plt.ylabel('Uobr')
    plt.plot(time_labels, mean, label="actual", marker="o")
    plt.plot(time_labels, predict_values, label="prediction", marker="o")
    plt.legend()
    plt.savefig(f"result/{u}/3.jpg")
    plt.close()

    model_std = LinearRegression()
    model_std.fit(time_labels_np, std)
    std_coef = model_std.coef_[0]
    predict_values = model_std.predict(time_labels_np)
    plt.figure(figsize=(11, 7))
    plt.title('Linear Regression (std)')
    plt.xlabel('time')
    plt.ylabel('Uobr')
    plt.plot(time_labels, std, label="actual", marker="o")
    plt.plot(time_labels, predict_values, label="prediction", marker="o")
    plt.legend()
    plt.savefig(f"result/{u}/4.jpg")
    plt.close()

    mean_test_time0 = test[columns[0]].mean()
    std_test_time0 = test[columns[0]].std()
    mean_test = []
    std_test = []
    for i in time_labels:
        mean_test.append(mean_test_time0 + mean_coef * i)
        std_test.append(std_test_time0 + std_coef * i)

    test_working_predict = []
    for i in mean_test:
        for j in std_test:
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
        print(p, e)
        test_error += pow((p - e) / e, 2)
    test_error = np.sqrt(test_error / len(test_working_predict))

    f.write(f"\n\nПроцент рабочих устройств\nВремя: Прогноз (Реальное значение)\n")
    for t, p, e in zip(time_labels, test_working_predict, test_working_experiment):
        f.write(f"{t}: {p} ({e})\n")
    f.write(f"Ошибка: {test_error}")

    plt.figure(figsize=(11, 7))
    sns.histplot(test[columns[6]], bins=10)
    x = np.linspace(mean_test[len(mean_test) - 1] - 4 * std_test[len(std_test) - 1], mean_test[len(mean_test) - 1] + 5 * std_test[len(std_test) - 1], 100)
    plt.plot(x, sps.norm(loc=mean_test[len(mean_test) - 1], scale=std_test[len(std_test) - 1]).pdf(x), color='darkorange')
    plt.axvline(bottom_limit, 0, 1, color='k')
    plt.axvline(top_limit, 0, 1, color='k')
    plt.savefig(f"result/{u}/5.jpg")
    plt.close()

    f.close()
