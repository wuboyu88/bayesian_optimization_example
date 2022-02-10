"""This python script illustrates how to implement Bayesian Optimization from Scratch in Python.
Please refer to https://distill.pub/2020/bayesian-optimization for details"""

__author__ = 'Boyu Wu'

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings
from warnings import simplefilter
import matplotlib.pyplot as plt


def sort_two_arrays_together(a, b):
    """
    同时对两个数组排序
    :param a: [1, 3, 2]
    :param b: [7, 9, 10]
    :return: [1, 2, 3], [7, 10, 9]
    """
    a_sorted, b_sorted = zip(*sorted(zip(a, b)))
    a_sorted = np.array(a_sorted)
    b_sorted = np.array(b_sorted)
    return a_sorted, b_sorted


def f(x):
    """
    目标函数
    :param x:
    :return: xsin(x)
    """
    return x * np.sin(x)


def surrogate(model, x):
    """
    目标函数对应的代理函数
    :param model: 高斯回归模型
    :param x: 待预测的点x
    :return: mu, std
    """
    with catch_warnings():
        simplefilter('ignore')
        return model.predict(x, return_std=True)


def acquisition(x_labeled, y_labeled, x_bounds, model, axs, n_iter, nb_axs, method='UCB', epsilon=0, alpha=1):
    """
    采集函数
    :param x_labeled: 训练集对应的x
    :param y_labeled: 训练集对应的y
    :param x_bounds: 待调的参数范围
    :param model: 高斯回归模型
    :param axs: subplot对应的axs
    :param n_iter: 迭代的次数
    :param nb_axs: 第nb_axs张图片
    :param method: 'PI' (probability of improvement) or 'UCB' (upper confidence bound)
    :param epsilon: 提升的幅度
    :param alpha: trade off between exploration and exploitation
    :return acquisition_score: 采集函数的分数
    """
    # 画出训练集对应的点
    x_sorted, y_sorted = sort_two_arrays_together(x_labeled, y_labeled)

    label = r'Training Points' if nb_axs == n_iter - 1 else None
    axs[nb_axs].scatter(x_sorted, y_sorted, marker='o', color='black', label=label)

    # 训练集对应的y最大值
    best = max(y_labeled)

    # 通过代理函数计算出均值和标准差
    mu, std = surrogate(model, x_bounds)
    mu = mu[:, 0]

    # 计算采集函数的值
    if method == 'PI':
        acquisition_score = norm.cdf((mu - best - epsilon) / (std + 1E-9))
    elif method == 'UCB':
        acquisition_score = mu + alpha * std
    else:
        print('You have to choose an implemented acquisition method!')
        return

    x_sorted, y_sorted = sort_two_arrays_together(x_bounds, acquisition_score)
    axs[nb_axs].plot(x_sorted, y_sorted, 'green')

    # 画出采集函数达到最大值对应的点
    ix = np.argmax(y_sorted)

    label = r'Query Point' if nb_axs == n_iter - 1 else None
    axs[nb_axs].scatter(x_sorted[ix, 0], y_sorted[ix], marker='o', color='red', s=80, label=label)
    return acquisition_score


def opt_acquisition(x_labeled, y_labeled, model, x_bounds, axs, nb_iter, nb_axs, method='UCB', epsilon=0, alpha=1):
    """
    得到采集函数达到最大值对应的点
    :param x_labeled: 训练集对应的x
    :param y_labeled: 训练集对应的y
    :param model: 高斯回归模型
    :param x_bounds: 待调的参数范围
    :param axs: subplot对应的axs
    :param nb_iter: 迭代的次数
    :param nb_axs: 第nb_axs张图片
    :param method: 'PI' (probability of improvement) or 'UCB' (upper confidence bound)
    :param epsilon: 提升的幅度
    :param alpha: trade off between exploration and exploitation
    :return query_x: 采集函数达到最大值对应的点
    """
    # 计算每个采集点对应的采集函数值
    acquisition_score = acquisition(x_labeled, y_labeled, x_bounds, model, axs, nb_iter, nb_axs, method, epsilon, alpha)

    # 得到采集函数达到最大值对应的点作为下一个待挖掘的点
    query_x = x_bounds[np.argmax(acquisition_score), 0]

    return query_x


def plot(model, x, y_true, axs, n_iter, nb_axs):
    """
    画图
    :param model: 高斯回归模型
    :param x: 真实目标函数的x
    :param y_true: 真实目标函数的y
    :param axs: subplot对应的axs
    :param n_iter: 迭代的次数
    :param nb_axs: 第nb_axs张图片
    :return:
    """
    # 画出真实函数对应的图像
    label = r'Ground Truth($f$)' if nb_axs == n_iter - 1 else None
    axs[nb_axs].plot(x[:, 0], y_true[:, 0], 'violet', label=label)

    # 画出代理函数对应的图像
    y_mu, y_std = surrogate(model, x)

    label = r'Predicted($\mu$)' if nb_axs == n_iter - 1 else None
    axs[nb_axs].plot(x, y_mu, 'black', label=label)

    # 画出代理函数对应的置信区间
    fit_up = y_mu[:, 0] + y_std
    fit_down = y_mu[:, 0] - y_std
    x_sorted, fit_up_sorted = sort_two_arrays_together(x, fit_up)
    x_sorted, fit_down_sorted = sort_two_arrays_together(x, fit_down)
    a = np.concatenate([x_sorted, x_sorted[::-1]])
    b = np.concatenate([fit_down_sorted, fit_up_sorted[::-1]])

    label = r'$\mu \pm \sigma$' if nb_axs == n_iter - 1 else None
    axs[nb_axs].fill(a, b, alpha=.8, facecolor='grey', label=label)

    axs[nb_axs].set_ylabel('Iteration: {}'.format(nb_axs))


def bayesian_optimization_example():
    """
    贝叶斯优化例子
    :return:
    """
    # 迭代的次数
    n_iter = 5
    fig, axs = plt.subplots(n_iter)

    # 真实目标函数图像需要的点
    x = np.linspace(0, 10, 1000)
    y_true = np.array([f(ele) for ele in x])
    x = x.reshape(len(x), 1)
    y_true = y_true.reshape(len(y_true), 1)

    # 初始训练集的点
    x_labeled = np.array([2, 7])
    y_labeled = np.array([f(ele) for ele in x_labeled])
    x_labeled = x_labeled.reshape(len(x_labeled), 1)
    y_labeled = y_labeled.reshape(len(y_labeled), 1)

    # 待调的参数范围
    np.random.seed(10)
    x_bounds = np.random.random(100) * 10
    x_bounds = x_bounds.reshape(len(x_bounds), 1)

    # 训练高斯回归模型
    gp_model = GaussianProcessRegressor()
    gp_model.fit(x_labeled, y_labeled)

    # 贝叶斯优化过程
    for i in range(n_iter):
        # 选取下一个待挖掘的点
        query_x = opt_acquisition(x_labeled, y_labeled, gp_model, x_bounds, axs, n_iter, nb_axs=i)

        # 得到该点对应的真实值（在实际情况中，通常这一步的代价比较大）
        ground_truth_f = f(query_x)

        # 打印优化的过程
        predicted_f, _ = surrogate(gp_model, [[query_x]])
        predicted_f = predicted_f[0][0]
        print('[iter={}] Query_x={}, Predicted_f={}, Ground_Truth_f={}'.format(i, round(query_x, 3),
                                                                               round(predicted_f, 3),
                                                                               round(ground_truth_f, 3)))
        # 画出每次优化对应的图
        plot(gp_model, x, y_true, axs, n_iter, nb_axs=i)

        # 将该点加入已知的数据集中
        x_labeled = np.vstack((x_labeled, [[query_x]]))
        y_labeled = np.vstack((y_labeled, [[ground_truth_f]]))

        x_labeled, y_labeled = sort_two_arrays_together(x_labeled, y_labeled)

        # 更新高斯回归模型
        gp_model.fit(x_labeled, y_labeled)

    fig.suptitle('Bayesian Optimization Process')
    handles, labels = axs[n_iter - 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', prop={'size': 6.5})


if __name__ == '__main__':
    bayesian_optimization_example()
