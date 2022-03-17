import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as sps
from scipy.misc import derivative
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def split3(l):
    l1, l2 = train_test_split(l, test_size=0.66, random_state=42)
    l2, l3 = train_test_split(l2, test_size=0.5, random_state=42)
    return [l1, l2, l3]


def weibull(train, test, columns, param, file_limits, weibull_shape, weibull_function=None, plot_path=None):
    print(param, weibull_shape)
    time_labels = [int(i.split("_")[1]) for i in columns]
    train_split3 = split3(train)
    limits = file_limits[file_limits["name"] == param]
    bottom_limit = float(limits["bottom_limit"])
    top_limit = float(limits["top_limit"])
    if weibull_shape is None:
        weibull_shape = limits["weibull_shape"].iloc[0]
    if weibull_function is None:
        weibull_function = limits["weibull_function"].iloc[0]
    if plot_path is not None:
        sns.set_style("darkgrid")
        os.makedirs(f"{plot_path}/3")

    weibull = sps.weibull_max if weibull_function == "max" else sps.weibull_min

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

    if plot_path is not None:
        plt.figure(figsize=(11, 7))
        plt.title('Linear Regression (loc)')
        plt.xlabel('time')
        plt.ylabel('loc')
        plt.plot(time_labels, [weibull.fit(train[i], fc=weibull_shape)[1] for i in columns], label="loc", marker="o")
        plt.plot(time_labels, model_loc.predict(linear_regression_x_train), label="predict_values", marker="o")
        plt.legend()
        plt.savefig(f"{plot_path}/1.jpg")
        plt.close()

        plt.figure(figsize=(11, 7))
        plt.title('Linear Regression (scale)')
        plt.xlabel('time')
        plt.ylabel('scale')
        plt.plot(time_labels, [weibull.fit(train[i], fc=weibull_shape)[2] for i in columns], label="scale", marker="o")
        plt.plot(time_labels, model_scale.predict(linear_regression_x_train), label="predict_values", marker="o")
        plt.legend()
        plt.savefig(f"{plot_path}/2.jpg")
        plt.close()

    test_mean0 = np.array(test[columns[0]]).mean()
    test_std0 = np.array(test[columns[0]]).std()
    linear_regression_x_test = np.array([[i, test_mean0, test_std0] for i in time_labels])
    loc_test = model_loc.predict(linear_regression_x_test)
    scale_test = model_scale.predict(linear_regression_x_test)

    if plot_path is not None:
        for i, j, k, l in zip(time_labels, columns, loc_test, scale_test):
            x = np.linspace(test[j].min() - 0.01, test[j].max() + 0.01, 100)
            plt.title(f'Time {i}')
            plt.hist(test[j])
            plt.plot(x, weibull(weibull_shape, k, l).pdf(x))
            plt.axvline(bottom_limit, 0, 1, color='k')
            plt.axvline(top_limit, 0, 1, color='k')
            plt.savefig(f"{plot_path}/3/{i}.jpg")
            plt.close()

    test_working_predict = []
    for loc, scale in zip(loc_test, scale_test):
        if loc < 0:
            loc = 0.01
        if scale < 0:
            scale = 0.01
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

    return time_labels, test_working_predict, test_working_experiment, test_error


def main():
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

    print("1 - 80/20 с графиками\n"
          "2 - Разные варианты тренировочной и контрольной выборок\n"
          "3 - График ошибок при разных значениях shape\n"
          "4 - Поиск оптимального shape с помощью градиентного спуска")
    o = input()
    if o == "1":
        path = "weibull_distribution/result1"
        if os.path.isdir(path):
            shutil.rmtree(path)
        train, test = train_test_split(file, test_size=0.2, random_state=42)
        for param in dict_columns:
            os.makedirs(f"{path}/{param}")
            weibull_shape = float(file_limits[file_limits["name"] == param]["weibull_shape"])
            time, predict, experiment, error = weibull(train, test, dict_columns[param], param, file_limits,
                                                       weibull_shape, None, f"{path}/{param}")
            f = open(f"{path}/{param}/results.txt", "w", encoding="utf-8")
            f.write(f"Процент рабочих устройств:\n{'Время'.ljust(8)}{'Прогноз'.ljust(21)}Реальное значение\n")
            for t, p, e in zip(time, predict, experiment):
                f.write(f"{str(t).ljust(8)}{str(p).ljust(21)}{e}\n")
            f.write(f"Ошибка: {error}")
            f.close()
    elif o == "2":
        path = "weibull_distribution/result2"
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.makedirs(path)
        for param in dict_columns:
            for dist in np.linspace(0.1, 0.5, 5):
                train, test = train_test_split(file, test_size=dist, random_state=42)
                weibull_shape = float(file_limits[file_limits["name"] == param]["weibull_shape"])
                time, predict, experiment, error = weibull(train, test, dict_columns[param], param, file_limits,
                                                           weibull_shape)
                f = open(f"{path}/{param}.txt", "a", encoding="utf-8")
                f.write(f"Обучающая выборка - {100 - int(dist * 100)}%, тестовая выборка - {int(dist * 100)}%\n")
                f.write(f"Процент рабочих устройств:\n{'Время'.ljust(8)}{'Прогноз'.ljust(21)}Реальное значение\n")
                for t, p, e in zip(time, predict, experiment):
                    f.write(f"{str(t).ljust(8)}{str(p).ljust(21)}{e}\n")
                f.write(f"Ошибка: {error}\n\n")
                f.close()
    elif o == "3":
        path = "weibull_distribution/result3"
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.makedirs(f"{path}")
        train, test = train_test_split(file, test_size=0.2, random_state=42)
        shapes = np.linspace(0.1, 10, 100)
        for param in dict_columns:
            weibull_shape_base = float(file_limits[file_limits["name"] == param]["weibull_shape"])
            errors1 = []
            for weibull_shape in shapes:
                time, predict, experiment, error = weibull(train, test, dict_columns[param], param, file_limits,
                                                           weibull_shape, "min")
                errors1.append(error)
            errors2 = []
            for weibull_shape in shapes:
                time, predict, experiment, error = weibull(train, test, dict_columns[param], param, file_limits,
                                                           weibull_shape, "max")
                errors2.append(error)
            plt.title(f'График ошибок {param} min')
            plt.xlabel('shape')
            plt.ylabel('error')
            plt.plot(shapes, errors1)
            plt.savefig(f"{path}/{param}_1.jpg")
            plt.close()
            plt.title(f'График ошибок {param} max')
            plt.xlabel('shape')
            plt.ylabel('error')
            plt.plot(shapes, errors2)
            plt.savefig(f"{path}/{param}_2.jpg")
            plt.close()
    elif o == "4":
        path = "weibull_distribution/result4"
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.makedirs(f"{path}")
        train, test = train_test_split(file, test_size=0.2, random_state=42)
        for param in dict_columns:
            shapes = np.linspace(0.1, 15, 20)
            errors = []
            for shape in shapes:
                time, predict, experiment, error = weibull(train, test, dict_columns[param], param, file_limits, shape)
                errors.append(error)

            def f(shape=None):
                if shape is not None and shape <= 0:
                    return 1
                return weibull(train, test, dict_columns[param], param, file_limits, shape)[3]

            def gradient_descent(curr_shape, prec, rate):

                descent_hist = {curr_shape: f(curr_shape)}
                prev_shape = curr_shape + prec + 1
                while abs(curr_shape - prev_shape) > prec:
                # while abs(curr_shape - prev_shape) > prec and len(descent_hist) < 50:
                    print(abs(curr_shape - prev_shape), len(descent_hist))
                    prev_shape = curr_shape
                    deriv = derivative(f, prev_shape, dx=0.01)
                    curr_shape = prev_shape - rate * deriv
                    descent_hist[curr_shape] = f(curr_shape)
                return curr_shape, descent_hist

            descent_hist = gradient_descent(2, 0.01, 1)[1]
            # descent_hist = gradient_descent(2, 0.01, 1)[1] | gradient_descent(4, 0.01, 1)[1] | gradient_descent(6, 0.01, 1)[1]
            for i, j in descent_hist.items():
                print(i, j)
            weibull_shape = min(descent_hist, key=descent_hist.get)
            print("Ошибка прогнозом", f(weibull_shape))
            print("Ошибка ручками", f())
            plt.title(f'График ошибок {param}')
            plt.xlabel('shape')
            plt.ylabel('error')
            plt.plot(shapes, errors)
            plt.plot(list(descent_hist.keys()), list(descent_hist.values()), 'ro')
            plt.axvline(weibull_shape)
            plt.xlim([0, 15])
            plt.ylim([0, 1])
            plt.savefig(f"{path}/{param}.jpg")
            plt.close()


if __name__ == "__main__":
    main()
