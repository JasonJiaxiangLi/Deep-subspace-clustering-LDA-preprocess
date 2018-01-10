# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 10:29:01 2017

@author: Sallysyw & Jasonljx
"""
import numpy as np
from scipy.optimize import minimize


# method in the original paper
def GET_C(X,size=50,gamma=0.1):
  A = np.transpose(X[0:size,:])
  fun = lambda x: np.linalg.norm(A*x.reshape((size,size)) - A,2)+gamma*np.linalg.norm(x,1)
  cons = ({'type': 'eq', 'fun': lambda x: np.diag(x.reshape((size,size)))==np.zeros(size)})
  res = minimize(fun, np.zeros((size,size)),method='SLSQP',constraints=cons)
  C = res.x.reshape((size,size))
  
  for i in range(size):
        for j in range(size):
                if abs(C[i,j])<1e-3:
                     C[i,j] = 0
  return np.transpose(C),res


# A class that determine the value of S and C in our model
class getcs:
    def __init__(self, data_mat, label, t, gamma):
        self.X = data_mat
        self.label = label
        self.t = t
        self.gamma = gamma
        Miu_i, miu = self.cal_miu()
        self.Miu_i = Miu_i
        self.miu = miu


    # note that miu_i contains means of different groups, miu is the overall average
    def cal_miu(self):
        dim = self.X.shape[1]
        N = self.X.shape[0]
        Miu_i = []
        label_set = []
        miu_i = np.zeros(dim)
        miu = np.zeros(dim)
        n_i = 0
        flag = self.label[0]
        for i in range(N):
            if self.label[i] == flag:
                n_i = n_i + 1
                miu_i = miu_i + self.X[i]
            else:
                Miu_i.append(miu_i / n_i)
                n_i = 0
                miu_i = np.zeros(dim)
                flag = self.label[i]
            miu = miu + self.X[i]
        Miu_i.append(miu_i / n_i)
        miu = miu / N
        return Miu_i, miu

    # calculate the objective function
    def obj_function(self, x):
        N = self.X.shape[0]
        d = self.X.shape[1]
        S = np.asmatrix(x[0:d * self.t].reshape((d, self.t)))
        C = np.asmatrix(x[d * self.t:d * self.t + N * N].reshape((N, N)))
        Miu_i = self.Miu_i
        n = len(Miu_i)

        # Compute term 1
        term1 = 0
        j = 0; k = 0
        temp = np.zeros((1, self.t))
        flag = self.label[j]
        for i in range(N):
            c_i = np.transpose(C)[i]
            if self.label[i] == flag:
                temp = temp + np.dot(c_i, np.dot(self.X, S))
                k = k + 1
            else:
                u = Miu_i[j]
                temp = np.dot(u.reshape(1, (np.size(u))), S) - temp / k
                term1 = term1 + np.linalg.norm(temp)
                temp = np.zeros((1, self.t))
                k = 0; j = j + 1
                flag = self.label[i]

        # Compute term2
        j = 0; k = 0
        flag = self.label[j]
        term2 = 0
        W = np.zeros((d, d))
        for i in range(N):
            if self.label[i] == flag:
                u = self.X[i] - Miu_i[j]
                W = W + np.dot(u.reshape((np.size(u), 1)), u.reshape(1, (np.size(u))))
                k = k + 1
            else:
                term2 = term2 + np.linalg.norm(np.dot(np.dot(np.transpose(S), W), S)) / k
                flag = self.label[i]
                W = np.zeros((d, d))
                j = j + 1; k = 0

        # Term 3
        term3 = self.gamma * np.linalg.norm(C, 1)

        # Add sum
        term = term1 / 1000 + term2 / 100000 + term3
        return term

    def cons_function1(self, x):
        N = self.X.shape[0]
        d = self.X.shape[1]
        C = np.asmatrix(x[d * self.t:d * self.t + N * N].reshape((N, N)))
        return np.linalg.norm(np.diag(C), 1)

    def cons_function2(self, x):
        d = self.X.shape[1]
        S = np.asmatrix(x[0:d * self.t].reshape((d, self.t)))
        B = np.zeros((d, d))
        Miu_i = self.Miu_i
        miu_mean = self.miu
        for miu in Miu_i:
            u = miu - miu_mean
            B = B + np.dot(u.reshape((np.size(u), 1)), u.reshape(1, (np.size(u))))
        term = np.linalg.norm(np.dot(np.dot(np.transpose(S), B), S) - np.eye(self.t), 1)
        return term

    def get_cs(self):
        N = self.X.shape[0]
        d = self.X.shape[1]
        t = self.t
        fun = lambda x: self.obj_function(x)
        cons = ({'type': 'eq', 'fun': lambda x: self.cons_function1(x)},
                {'type': 'eq', 'fun': lambda x: self.cons_function2(x)})
        x_0 = np.random.rand(d * t + N * N)  # initial value
        # x_0 = np.zeros(d * t + N * N)
        res = minimize(fun, x_0, method='SLSQP', constraints=cons)

        S = np.asmatrix(res.x[0:d * t].reshape((d, t)))
        C = np.asmatrix(res.x[d * t:d * t + N * N].reshape((N, N)))

        for i in range(N):
            for j in range(N):
                if abs(C[i, j]) < 1e-3:
                    C[i, j] = 0

        return C, S