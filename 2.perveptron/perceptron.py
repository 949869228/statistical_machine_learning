import numpy as np


class Perception:

    def __init__(self, lr=0.1, max_iter=100, random_state=43):
        self.lr = lr
        self.max_iter = max_iter
        self.w = None  # (n_fea, )
        self.b = None
        np.random.seed(random_state)

    def fit(self, x, y):
        m, n = x.shape
        self.w = np.random.random(n)
        self.b = np.random.random()
        exist_error = True
        iteration = 0
        while exist_error and iteration <= self.max_iter:
            i = np.random.randint(m)
            x_tmp = x[i]
            y_tmp = y[i]
            if y_tmp * (np.dot(x_tmp, self.w) + self.b) <= 0:
                self.w += self.lr * y_tmp * x_tmp
                self.b += self.lr * y_tmp
            exist_error = np.sum((np.dot(x, self.w) + self.b) * y <= 0)
            iteration += 1
        print("iteration times:", iteration)

    def predict(self, x):
        pred = np.dot(x, self.w) + self.b
        return self.__sign(pred)

    def __sign(self, x):
        return (x >= 0).astype(int) - (x < 0).astype(int)


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    iris = load_iris()
    x = iris.data[:100]
    y = iris.target[:100]
    y = 2 * y - 1
    clf = Perception()
    clf.fit(x, y)
    y_pred = clf.predict(x)
    print("accuracy=", np.sum(y_pred == y) / len(y))
