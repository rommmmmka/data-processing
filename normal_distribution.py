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
if os.path.isdir("normal_distribution/result"):
    shutil.rmtree("normal_distribution/result")

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
    PATH = f"normal_distribution/result/{u}"
    columns = dict_columns[u]
    time_labels = [int(i.split("_")[1]) for i in columns]
    limits = file_limits[file_limits["name"] == u]
    bottom_limit = float(limits["bottom_limit"])
    top_limit = float(limits["top_limit"])
    os.makedirs(f"{PATH}/7")
    f = open(f"{PATH}/results.txt", "w", encoding="utf-8")

    plt.figure(figsize=(11, 7))
    for i in file[columns].iloc:
        data = np.array(i)
        plt.xlabel('time')
        plt.ylabel(u)
        plt.plot(time_labels, data, color='steelblue', linewidth=1)
    plt.axhline(y=bottom_limit, xmin=0, xmax=1, color='k', linewidth=1.5)
    plt.axhline(y=top_limit, xmin=0, xmax=1, color='k', linewidth=1.5)
    plt.savefig(f"{PATH}/1.jpg")
    plt.close()

    plt.figure(figsize=(11, 7))
    for i in train[columns].iloc:
        data = np.array(i)
        plt.title('Train data')
        plt.xlabel('time')
        plt.ylabel(u)
        plt.plot(time_labels, data, color='steelblue', linewidth=1)
    plt.axhline(y=bottom_limit, xmin=0, xmax=1, color='k', linewidth=1.5)
    plt.axhline(y=top_limit, xmin=0, xmax=1, color='k', linewidth=1.5)
    plt.savefig(f"{PATH}/2.jpg")
    plt.close()

    plt.figure(figsize=(11, 7))
    for i in test[columns].iloc:
        data = np.array(i)
        plt.title('Test data')
        plt.xlabel('time')
        plt.ylabel(u)
        plt.plot(time_labels, data, color='steelblue', linewidth=1)
    plt.axhline(y=bottom_limit, xmin=0, xmax=1, color='k', linewidth=1.5)
    plt.axhline(y=top_limit, xmin=0, xmax=1, color='k', linewidth=1.5)
    plt.savefig(f"{PATH}/3.jpg")
    plt.close()

    plt.figure(figsize=(11, 7))
    for i in file[columns].iloc:
        data = np.array(i)
        plt.xlabel('time')
        plt.ylabel(u)
        plt.scatter(time_labels, data, color='steelblue', marker=".")
    plt.axhline(y=bottom_limit, xmin=0, xmax=1, color='k', linewidth=1.5)
    plt.axhline(y=top_limit, xmin=0, xmax=1, color='k', linewidth=1.5)
    plt.savefig(f"{PATH}/4.jpg")
    plt.close()

    linear_regression_x = []
    mean = []
    std = []
    for i in train_split3:
        train_mean0 = np.array(i[columns[0]]).mean()
        train_std0 = np.array(i[columns[0]]).std()
        for t, c in zip(time_labels, columns):
            linear_regression_x.append([t, train_mean0, train_std0])
            mean.append(np.array(i[c]).mean())
            std.append(np.array(i[c]).std())
    linear_regression_x = np.array(linear_regression_x)

    model_mean = LinearRegression()
    model_mean.fit(linear_regression_x, mean)
    model_std = LinearRegression()
    model_std.fit(linear_regression_x, std)

    train_mean0 = np.array(train[columns[0]]).mean()
    train_std0 = np.array(train[columns[0]]).std()
    linear_regression_x_train = np.array([[i, train_mean0, train_std0] for i in time_labels])

    plt.figure(figsize=(11, 7))
    plt.title('Linear Regression (mean)')
    plt.xlabel('time')
    plt.ylabel('mean')
    plt.plot(time_labels, [np.array(train[i]).mean() for i in columns], label="mean", marker="o")
    plt.plot(time_labels, model_mean.predict(linear_regression_x_train), label="predict_values", marker="o")
    plt.legend()
    plt.savefig(f"{PATH}/5.jpg")
    plt.close()

    plt.figure(figsize=(11, 7))
    plt.title('Linear Regression (std)')
    plt.xlabel('time')
    plt.ylabel('std')
    plt.plot(time_labels, [np.array(train[i]).std() for i in columns], label="std", marker="o")
    plt.plot(time_labels, model_std.predict(linear_regression_x_train), label="predict_values", marker="o")
    plt.legend()
    plt.savefig(f"{PATH}/6.jpg")
    plt.close()

    test_mean0 = np.array(test[columns[0]]).mean()
    test_std0 = np.array(test[columns[0]]).std()
    linear_regression_x_test = np.array([[i, test_mean0, test_std0] for i in time_labels])
    mean_test = model_mean.predict(linear_regression_x_test)
    std_test = model_std.predict(linear_regression_x_test)

    for n, (i, j, k, l) in enumerate(zip(time_labels, columns, mean_test, std_test)):
        x = np.linspace(k - 4 * l, k + 4 * l, 100)
        plt.title(f'Time {i}')
        plt.hist(test[j])
        plt.plot(x, sps.norm(loc=k, scale=l).pdf(x), color='darkorange')
        plt.axvline(bottom_limit, 0, 1, color='k')
        plt.axvline(top_limit, 0, 1, color='k')
        plt.savefig(f"{PATH}/7/{i}.jpg")
        plt.close()

    test_working_predict = []
    for i, j in zip(mean_test, std_test):
        test_working_predict.append(sps.norm(loc=i, scale=j).cdf(top_limit) - sps.norm(loc=i, scale=j).cdf(bottom_limit))

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

    f.write(f"?????????????? ?????????????? ??????????????????:\n{'??????????'.ljust(8)}{'??????????????'.ljust(21)}???????????????? ????????????????\n")
    for t, p, e in zip(time_labels, test_working_predict, test_working_experiment):
        f.write(f"{str(t).ljust(8)}{str(p).ljust(21)}{e}\n")
    f.write(f"????????????: {test_error}")

    f.close()
