"""Use tensorflow constrained optimization

@author: Sallysyw & Jasonljx
"""
import numpy as np
from scipy.optimize import minimize
import tensorflow as tf

learning_rate = 0.1
max_steps = 10000
display_step = 10
penalty_increase_factor = 1.00021

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


# Use gradient descent to get C and S, through tensorflow
class tf_getcs:
    def __init__(self, data_mat, label, t, gamma):
        self.X = data_mat
        self.label = label
        self.t = t
        self.gamma = gamma
        Miu_i, miu = self.cal_miu()
        self.Miu_i = Miu_i
        self.miu = miu
        dim = self.X.shape[1]
        # Calculate the inter-group difference matrix B
        B = np.zeros((dim, dim))
        for miu in self.Miu_i:
            u = miu - self.miu
            B = B + np.dot(u.reshape((np.size(u), 1)), u.reshape(1, (np.size(u))))
        self.B = B


    # note that miu_i contains means of different groups, miu is the overall average
    def cal_miu(self):
        dim = self.X.shape[1]
        N = self.X.shape[0]
        Miu_i = []
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
    def loss(self, x, lambda_con):
        X = tf.convert_to_tensor(self.X, dtype=tf.float32)
        N = int(X.shape[0])
        d = int(X.shape[1])
        t = self.t

        Miu_i =[tf.convert_to_tensor(miu_i, dtype=tf.float32) for miu_i in self.Miu_i]

        S = tf.reshape(x[0:d * t], [d, t])
        C = tf.reshape(x[d * t:d * t + N * N], [N, N])

        # Compute term 1
        term1 = 0
        j = 0; k = 0
        temp = tf.zeros([1, t])
        flag = self.label[j]
        for i in range(N):
            c_i = tf.reshape(tf.transpose(C, [1, 0])[i], [1, N])
            if self.label[i] == flag:
                temp = temp + tf.matmul(c_i, tf.matmul(X, S))
                k = k + 1
            else:
                u = Miu_i[j]
                u = tf.cast(tf.reshape(u, [1, int(u.get_shape()[0])]), dtype=tf.float32)
                temp = tf.matmul(u, S) - temp / k
                term1 = term1 + tf.norm(temp)
                temp = tf.zeros((1, t))
                k = 0; j = j + 1
                flag = self.label[i]

        # Compute term2
        j = 0; k = 0
        flag = self.label[j]
        term2 = 0
        W = tf.zeros([d, d])
        for i in range(N):
            if self.label[i] == flag:
                u = X[i] - Miu_i[j]
                W = W + tf.matmul(tf.reshape(u, [int(u.get_shape()[0]), 1]), tf.reshape(u, [1, int(u.get_shape()[0])]))
                k = k + 1
            else:
                term2 = term2 + tf.norm(tf.matmul(tf.matmul(S, W, transpose_a=True), S)) / k
                flag = self.label[i]
                W = tf.zeros([d, d])
                j = j + 1; k = 0

        # Term 3
        term3 = self.gamma * tf.reduce_max(tf.abs(C))

        cons1 = self.cons_function1(C)
        cons2 = self.cons_function2(S)

        # Penalty function
        term = (term1 / d + term2 / (N * d) + term3 + cons1*lambda_con[0] + cons2*lambda_con[1] / (N * N)) / d

        # return term1, term2, term3, cons1, cons2, term
        return term

    def cons_function1(self, C):
        return tf.reduce_max(tf.abs(tf.diag_part(C)))

    def cons_function2(self, S):
        B = tf.convert_to_tensor(self.B, dtype=tf.float32)
        temp = tf.matmul(tf.matmul(S, B, transpose_a=True), S) - tf.convert_to_tensor(np.eye(self.t), dtype=tf.float32)
        term = tf.reduce_max(tf.abs(temp))
        return term

    def training(self, total_loss, global_step):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(total_loss, global_step=global_step)
        return train_op

    def get_cs(self):
        N = self.X.shape[0]
        d = self.X.shape[1]
        t = self.t

        x_tf = tf.get_variable(name='x', shape=[d * t + N * N],
                               initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
        lambda_np = [1, 2]
        lambda_con = [tf.Variable(lambda_np[0], dtype=tf.float32), tf.Variable(lambda_np[1], dtype=tf.float32)]
        global_step = tf.contrib.framework.get_or_create_global_step()

        loss = self.loss(x_tf, lambda_con=lambda_con)

        train_op = self.training(loss, global_step)

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            print("Use penalty method to get C and S")
            sess.run(init)
            for i in range(max_steps):
                _, loss_value = sess.run([train_op, loss])
                if i % display_step == 0:
                    print("Epoch %d, loss %.5f" % (i, loss_value))
                lambda_np = [lambda_i * penalty_increase_factor for lambda_i in lambda_np]
                for j in range(2):
                    sess.run(tf.assign(lambda_con[j], lambda_np[j]))
            x = x_tf

            S = tf.reshape(x[0:d * t], [d, t])
            C = tf.reshape(x[d * t:d * t + N * N], [N, N])

            #for i in range(N):
            #    for j in range(N):
            #        if abs(C[i, j]) < 1e-3:
            #            C[i, j] = 0

            # Verification
            cons1, cons2 = sess.run([self.cons_function1(C), self.cons_function2(S)])
            print("Here we verify the two constraints: c1(x) = %.5f, c2(x) = %.5f"
                  % (cons1, cons2))

            S, C = sess.run([S, C])

        return C, S