"""
@ Xingzhi Sun
June 26 2021
Class definitions of logistic regression.
Instead of aggregating gradients, aggregate weights.
Guest
- Owns a partial dataset.
- Computes gradient, then update weight, then send weight to arbiter.
- Receives the aggregated weight from the arbiter, then update the weight.

Arbiter:
- Receives weight from guests.
- Checks convergence? or let the guests check.
- Aggregates the weights.
- Sends the aggregated weights to guests.
"""
import numpy as np
import warnings
class Guest:
    def __init__(self, X, y, learning_rate=0.01, use_linear_grad_loss=False):
        self.X_, self.y_ = X, y
        self.N_, self.p_ = X.shape
        self.w_ = np.zeros(self.p_)
        self.learning_rate_ = learning_rate
        self.use_linear_grad_loss_ = use_linear_grad_loss
    def loss(self):
        wx = self.X_ @ self.w_
        if self.use_linear_grad_loss_:
            loss_points = np.log(2.) - self.y_ * wx + 0.125 * self.y_ * wx*wx + 0.5 * wx
        else:
            sig = 1. / (1. + np.exp(-wx))
            loss_points = - (self.y_ * np.log(sig) + (1 - self.y_) * np.log(1 - sig))
        return loss_points.mean()
    def gradient(self):
        wx = self.X_ @ self.w_
        if self.use_linear_grad_loss_:
            return self.X_.T @ (0.5 - 0.25 * wx - self.y_) / self.N_
        else:
            sig = 1 / (1 + np.exp(-wx))
            return self.X_.T @ (sig - self.y_) / self.N_
    def send_weight(self):
        grad = self.gradient()
        print("Gradient:{}".format(grad))
        self.w_ -= self.learning_rate_ * grad
        return self.w_
    def update(self, grad):
        self.w_ -= self.learning_rate_ * grad
    def update_weight(self, weight):
        self.w_ = weight

class Arbiter:
    def __init__(self, X_list, y_list, learning_rate=0.01, use_linear_grad_loss=False, averaging_method="sample_size"):
        self.use_linear_grad_loss_ = use_linear_grad_loss
        self.n_parties_ = len(X_list)
        if len(y_list) != self.n_parties_:
            raise ValueError("len(X_list) should be equal to len(y_list)!")
        self.guests_ = []
        for i in range(len(X_list)):
            self.guests_.append(Guest(X_list[i], y_list[i], learning_rate, self.use_linear_grad_loss_))
        self.convergence_ratio_ = 0.
        self.max_iterations_ = 0
        if averaging_method == "sample_size":
            self.guest_weights_ = []
            for y_ in y_list:
                self.guest_weights_.append(y_.shape[0])
            self.guest_weights_ = np.array(self.guest_weights_)
            self.guest_weights_ = self.guest_weights_/self.guest_weights_.sum()
        else:
            raise ValueError("Invalid argument averaging_method: \"{}\"!".format(averaging_method))
    def convergence_check(self, last_loss, this_loss, iter_num):
        if iter_num >= self.max_iterations_:
            warnings.warn("Jumped out of loop at max iter! Did not converge. Max iter set too small?")
            return True
        delta = np.abs(last_loss - this_loss) / np.max([this_loss, 1.])
        if delta <= self.convergence_ratio_:
            print("Converged at delta = {}, converge_ratio={}".format(delta, self.convergence_ratio_))
            return True
        return False
    def aggregate_params(self, grad_list):
        return self.guest_weights_ @ np.array(grad_list)
    def train(self, converge_ratio=0.01, max_iterations=1000):
        self.convergence_ratio_ = converge_ratio
        self.max_iterations_ = max_iterations
        last_loss = np.Infinity
        is_converged = False
        round_num = 0
        while not is_converged:
            this_loss_list = []
            weight_list = []
            for i in range(self.n_parties_):
                weighti = self.guests_[i].send_weight()
                weight_list.append(weighti)
                print("Guest {}, weight:{}".format(i, np.round(self.guests_[i].w_, 3)))
            weight_aggregated = self.aggregate_params(weight_list)
            print("Aggregated weight:{}".format(np.round(weight_aggregated, 3)))
            for i in range(self.n_parties_):
                self.guests_[i].update_weight(weight_aggregated)
                this_loss_list.append(self.guests_[i].loss())
            this_loss = self.guest_weights_ @ np.array(this_loss_list)
            is_converged = self.convergence_check(last_loss, this_loss, round_num)
            print("Iter:{}, loss:{}, last_loss:{}".format(round_num, this_loss, last_loss))
            round_num += 1
            last_loss = this_loss
    def get_model(self):
        return LR_model(self.guests_[0].w_)

def add_const_col(X):
    return np.c_[np.ones(X.shape[0]), X]

class LR_model:
    def __init__(self, w):
        self.coef_ = w
    def set_coef(self, w):
        self.coef_ = w
    def predict(self, X):
        return (X @ self.coef_ > 0).astype(int)