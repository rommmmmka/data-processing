import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as sps
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

sns.set_style("darkgrid")
if os.path.isdir("weibull_distribution/result"):
    shutil.rmtree("weibull_distribution/result")

def split3(l):
    l1, l2 = train_test_split(l, test_size=0.66, random_state=42)
    l2, l3 = train_test_split(l2, test_size=0.5, random_state=42)
    return [l1, l2, l3]

file = pd.read_csv("data.csv")
file_limits = pd.read_csv("limits.csv")
train, test = train_test_split(file, test_size=0.2, random_state=42)
train_split3 = split3(train)

dict_columns = {}
for i, val in enumerate(file.columns.tolist()):
    if i == 0:
        continue
    data = val.split("_")
    if dict_columns.get(data[0]) is None:
        dict_columns[data[0]] = [val]
    else:
        dict_columns.get(data[0]).append(val)

for u in dict_columns:
    print(u)
    PATH = f"weibull_distribution/result/{u}"
    columns = dict_columns[u]
    time_labels = [int(i.split("_")[1]) for i in columns]
    limits = file_limits[file_limits["name"] == u]
    bottom_limit = float(limits["bottom_limit"])
    top_limit = float(limits["top_limit"])
    weibull_shape = float(limits["weibull_shape"])
    os.makedirs(f"{PATH}/3")
    f = open(f"{PATH}/results.txt", "w", encoding="utf-8")

    weibull = sps.weibull_max if limits["weibull_function"].iloc[0] == "max" else sps.weibull_min

    linear_regression_x = []
    loc = []
    scale = []
    for i in train_split3:
        train_mean0 = np.array(i[columns[0]]).mean()
        train_std0 = np.array(i[columns[0]]).std()
        for t, c in zip(time_labels, columns):
            linear_regression_x.append([t, train_mean0, train_std0])
            _shape, _loc, _scale = weibull.fit(i[c], fc=weibull_shape)
            loc.append(_loc)
            scale.append(_scale)
    linear_regression_x = np.array(linear_regression_x)

    model_loc = LinearRegression()
    model_loc.fit(linear_regression_x, loc)

    model_scale = LinearRegression()
    model_scale.fit(linear_regression_x, scale)

    train_mean0 = np.array(train[columns[0]]).mean()
    train_std0 = np.array(train[columns[0]]).std()
    linear_regression_x_train = np.array([[i, train_mean0, train_std0] for i in time_labels])

    plt.figure(figsize=(11, 7))
    plt.title('Linear Regression (loc)')
    plt.xlabel('time')
    plt.ylabel('loc')
    plt.plot(time_labels, [weibull.fit(train[i], fc=weibull_shape)[1] for i in columns], label="loc", marker="o")
    plt.plot(time_labels, model_loc.predict(linear_regression_x_train), label="predict_values", marker="o")
    plt.legend()
    plt.savefig(f"{PATH}/1.jpg")
    plt.close()

    plt.figure(figsize=(11, 7))
    plt.title('Linear Regression (scale)')
    plt.xlabel('time')
    plt.ylabel('scale')
    plt.plot(time_labels, [weibull.fit(train[i], fc=weibull_shape)[2] for i in columns], label="scale", marker="o")
    plt.plot(time_labels, model_scale.predict(linear_regression_x_train), label="predict_values", marker="o")
    plt.legend()
    plt.savefig(f"{PATH}/2.jpg")
    plt.close()

    test_mean0 = np.array(test[columns[0]]).mean()
    test_std0 = np.array(test[columns[0]]).std()
    linear_regression_x_test = np.array([[i, test_mean0, test_std0] for i in time_labels])
    loc_test = model_loc.predict(linear_regression_x_test)
    scale_test = model_scale.predict(linear_regression_x_test)

    for i, j, k, l in zip(time_labels, columns, loc_test, scale_test):
        x = np.linspace(test[j].min() - 0.01, test[j].max() + 0.01, 100)
        plt.title(f'Time {i}')
        plt.hist(test[j])
        plt.plot(x, weibull(weibull_shape, k, l).pdf(x))
        plt.axvline(bottom_limit, 0, 1, color='k')
        plt.axvline(top_limit, 0, 1, color='k')
        plt.savefig(f"{PATH}/3/{i}.jpg")
        plt.close()

    test_working_predict = []
    for loc, scale in zip(loc_test, scale_test):
        top = weibull.cdf(x=top_limit, c=weibull_shape, loc=loc, scale=scale)
        bottom = weibull.cdf(x=bottom_limit, c=weibull_shape, loc=loc, scale=scale)
        test_working_predict.append(top - bottom)

    test_working_experiment = []
    for i in columns:
        working = 0
        for j in test[i]:
            if bottom_limit < j < top_limit:
                working += 1
        test_working_experiment.append(working / test.shape[0])

    test_error = 0
    for p, e in zip(test_working_predict, test_working_experiment):
        test_error += pow((p - e) / e, 2)
    test_error = np.sqrt(test_error / len(test_working_predict))

    f.write(f"Процент рабочих устройств:\n{'Время'.ljust(8)}{'Прогноз'.ljust(21)}Реальное значение\n")
    for t, p, e in zip(time_labels, test_working_predict, test_working_experiment):
        f.write(f"{str(t).ljust(8)}{str(p).ljust(21)}{e}\n")
    f.write(f"Ошибка: {test_error}")

    f.close()
