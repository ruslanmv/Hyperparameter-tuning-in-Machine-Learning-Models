#! usr/bin/env python
# -*- coding : utf-8 -*-

import codecs
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, make_scorer
import time
import numpy as np
np.random.seed(123)
from skopt import gp_minimize
import matplotlib.pyplot as plt
from random import uniform
from skopt.acquisition import gaussian_ei


def main():
    # import some data to play with
    X = []
    y = []
    with codecs.open("../data/machine.data", 'r', 'utf-8') as infile:
        for line in infile:
            tokens = line.split(',')
            X.append([float(x) for x in tokens[:5]])
            y.append(float(tokens[6]))
    slice = int(round(len(X)*0.8))
    X_train = X[:slice]
    X_test = X[slice:]
    y_train = y[:slice]
    y_test = y[slice:]
    regr = linear_model.Lasso()
    regr.fit(X_train, y_train)
    y_predict = [i for i in regr.predict(X_test)]
    print("loss of the model:{}".format(mean_squared_error(y_test, y_predict)))

    #  apply gridsearch
    worst_case = float("inf")
    mse_gs_scores = []
    t0 = time.time()
    for g in [(i+1)*0.001 for i in range(8000)]:
        regr = linear_model.Lasso(alpha=g)
        regr.fit(X_train, y_train)
        y_pred = [i for i in regr.predict(X_test)]
        mse = mean_squared_error(y_test, y_pred)
        mse_gs_scores.append([g,mse])
        # save if best
        if mse < worst_case:
            worst_case = mse
            best_grid = g
    t1 = time.time()
    print("time taken by gridserach: {}".format(t1 - t0))
    print((worst_case,best_grid))

    # applying random search
    worst_case = float("inf")
    mse_rs_scores = []
    t0 = time.time()
    for _ in range(1000):
        g = uniform(0, 8)
        regr = linear_model.Lasso(alpha=g)
        regr.fit(X_train, y_train)
        y_pred = [i for i in regr.predict(X_test)]
        mse = mean_squared_error(y_test, y_pred)
        mse_rs_scores.append([g, mse])
        # save if best
        if mse < worst_case:
            worst_case = mse
            best_random = g
    t1 = time.time()
    print("time taken by randomserach: {}".format(t1 - t0))
    print((worst_case,best_random))

    # apply bayesian optimization
    noise_level = 0.1
    def f(alphavalue):
        regr = linear_model.Lasso(alpha=alphavalue)
        regr.fit(X_train, y_train)
        y_pred = [i for i in regr.predict(X_test)]
        return mean_squared_error(y_test, y_pred)
    x = np.array([(i+1)*0.001 for i in range(8000)])
    fx = [f(x_i) for x_i in x]
    plt.plot(x, fx, "r--", label="True (unknown)")
    plt.fill(np.concatenate([x, x[::-1]]),
             np.concatenate(([fx_i - 1.9600 * noise_level for fx_i in fx],
                             [fx_i + 1.9600 * noise_level for fx_i in fx[::-1]])),
             alpha=.2, fc="r", ec="None")
    t4 = time.time()
    res = gp_minimize(f,  # the function to minimize
                      [(0.001, 8.0)],  # the bounds on each dimension of x
                      acq_func="EI",  # the acquisition function
                      n_calls=15,  # the number of evaluations of f
                      n_random_starts=5,  # the number of random initialization points
                      random_state=123)
    t5 = time.time()
    print("time taken by BO_search: {}".format(t5 - t4))
    print(res['fun'])
    print(res['x'])

    plt.plot(res.x_iters, res.func_vals, "b--", label="BO")
    plt.plot([i[0] for i in mse_rs_scores][:10], [i[1] for i in mse_rs_scores][:10], "g--", label="Random Search")
    plt.legend()
    plt.grid()
    plt.show()




    plt.rcParams["figure.figsize"] = (8, 14)

    x = np.linspace(0.001, 8.0, 8000).reshape(-1, 1)
    x_gp = res.space.transform(x.tolist())
    fx = np.array([f(x_i) for x_i in x])

    # Plot the 5 iterations following the 5 random points
    for n_iter in range(5):
        gp = res.models[n_iter]
        curr_x_iters = res.x_iters[:5 + n_iter]
        curr_func_vals = res.func_vals[:5 + n_iter]

        # Plot true function.
        plt.subplot(5, 2, 2 * n_iter + 1)
        plt.plot(x, fx, "r--", label="True (unknown)")
        plt.fill(np.concatenate([x, x[::-1]]),
                 np.concatenate([fx - 1.9600 * noise_level,
                                 fx[::-1] + 1.9600 * noise_level]),
                 alpha=.2, fc="r", ec="None")

        # Plot GP(x) + contours
        y_pred, sigma = gp.predict(x_gp, return_std=True)
        plt.plot(x, y_pred, "g--", label=r"$\mu_{GP}(x)$")
        plt.fill(np.concatenate([x, x[::-1]]),
                 np.concatenate([y_pred - 1.9600 * sigma,
                                 (y_pred + 1.9600 * sigma)[::-1]]),
                 alpha=.2, fc="g", ec="None")

        # Plot sampled points
        plt.plot(curr_x_iters, curr_func_vals,
                 "r.", markersize=8, label="Observations")

        # Adjust plot layout
        plt.grid()

        if n_iter == 0:
            plt.legend(loc="best", prop={'size': 6}, numpoints=1)

        if n_iter != 4:
            plt.tick_params(axis='x', which='both', bottom='off',
                            top='off', labelbottom='off')

            # Plot EI(x)
        plt.subplot(5, 2, 2 * n_iter + 2)
        acq = gaussian_ei(x_gp, gp, y_opt=np.min(curr_func_vals))
        plt.plot(x, acq, "b", label="EI(x)")
        plt.fill_between(x.ravel(), -2.0, acq.ravel(), alpha=0.3, color='blue')

        next_x = res.x_iters[5 + n_iter]
        next_acq = gaussian_ei(res.space.transform([next_x]), gp, y_opt=np.min(curr_func_vals))
        plt.plot(next_x, next_acq, "bo", markersize=6, label="Next query point")

        # Adjust plot layout
        plt.ylim(0, 0.1)
        plt.grid()

        if n_iter == 0:
            plt.legend(loc="best", prop={'size': 6}, numpoints=1)

        if n_iter != 4:
            plt.tick_params(axis='x', which='both', bottom='off',
                            top='off', labelbottom='off')

    plt.show()

if __name__ == '__main__':
    main()
