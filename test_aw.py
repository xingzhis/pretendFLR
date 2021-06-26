"""
June 26 2021
test logistic regression.
"""
# %%
import numpy as np
from pFLR_aw import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
# %%
"""
Test one guest.
"""
np.random.seed(42)
N = 2000
def gt(v):
    return 1.0 * v[0] + 2.0 * v[1]
X = np.random.rand(N, 2)
y = np.apply_along_axis(gt, 1, X) + np.random.normal(0, 0.1, N)
y_med = np.median(y)
y_class = (y > y_med).astype(int)
X_1 = add_const_col(X)
X_train, X_test, y_train, y_test = train_test_split(X_1, y_class, test_size=0.2, random_state=42)

arb = Arbiter([X_train], [y_train], learning_rate=10, use_linear_grad_loss=False)
arb.train(converge_ratio=1e-5, max_iterations=10000)

myLR = arb.get_model()

LR = LogisticRegression()
LR.fit(X_train, y_train)
LRpred = LR.predict(X_test)
myLRpred = myLR.predict(X_test)
myLRdiff = ((LRpred - myLRpred) * (LRpred - myLRpred)).sum()
print("{} samples, {} different from sklearn LR, in average{}.".format(X_test.shape[0], myLRdiff, myLRdiff / X_test.shape[0]))
truediff = ((y_test - myLRpred) * (y_test - myLRpred)).sum()
print("{} samples, {} different from test true, in average{}.".format(X_test.shape[0], truediff, truediff / X_test.shape[0]))
LRdiff = ((y_test - LRpred) * (y_test - LRpred)).sum()
print("{} samples, {} sklearn LR different from test true, in average{}.".format(X_test.shape[0], LRdiff, LRdiff / X_test.shape[0]))
# %%
"""
Test multiple guests.
"""
np.random.seed(42)
N = 2000
def gt(v):
    return 1.0 * v[0] + 2.0 * v[1]
X = np.random.rand(N, 2)
y = np.apply_along_axis(gt, 1, X) + np.random.normal(0, 0.1, N)
y_med = np.median(y)
y_class = (y > y_med).astype(int)
X_1 = add_const_col(X)
X_train, X_test, y_train, y_test = train_test_split(X_1, y_class, test_size=0.2, random_state=42)

X1, X2, y1, y2 = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
X1, X3, y1, y3 = train_test_split(X1, y1, test_size=0.3, random_state=42)
X1, X4, y1, y4 = train_test_split(X1, y1, test_size=0.4, random_state=42)

arb = Arbiter([X1, X2, X3, X4], [y1, y2, y3, y4], learning_rate=10)
arb.train(converge_ratio=1e-5, max_iterations=10000)

myLR = arb.get_model()

LR = LogisticRegression()
LR.fit(X_train, y_train)
LRpred = LR.predict(X_test)
myLRpred = myLR.predict(X_test)
myLRdiff = ((LRpred - myLRpred) * (LRpred - myLRpred)).sum()
print("{} samples, {} different from sklearn LR, in average{}.".format(X_test.shape[0], myLRdiff, myLRdiff / X_test.shape[0]))
truediff = ((y_test - myLRpred) * (y_test - myLRpred)).sum()
print("{} samples, {} different from test true, in average{}.".format(X_test.shape[0], truediff, truediff / X_test.shape[0]))
LRdiff = ((y_test - LRpred) * (y_test - LRpred)).sum()
print("{} samples, {} sklearn LR different from test true, in average{}.".format(X_test.shape[0], LRdiff, LRdiff / X_test.shape[0]))

# %%
