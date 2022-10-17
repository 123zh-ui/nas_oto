import torch
import cvxpy as cvx
import numpy as np


__all__ = ["LASSO"]


class LASSO:
    def __init__(self, A):
        m, n = A.shape
        self.A = A
        self.m = m
        self.n = n
#等式9
    def construct_prob(self, b):
        gamma = cvx.Parameter(nonneg=True, value=1e-5)
        x = cvx.Variable(self.n)#初始的时候x.value是none
        error = cvx.sum_squares(self.A * x - b)
        obj = cvx.Minimize(0.5 * error + gamma * cvx.norm(x, 1))
        prob = cvx.Problem(obj)#Compute the gradient of a solution with respect to Parameters.

        return prob, x

    def solve(self, b):
        prob, x = self.construct_prob(b)#x.value None
        prob.solve(solver=cvx.MOSEK)#X.value进行更新,其和prob._solution一样
        x_res = np.array(x.value).reshape(-1)#，也就是对应的更新后重新获得的z
        return x_res
