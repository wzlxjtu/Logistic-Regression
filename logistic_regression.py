import numpy as np

np.random.seed(12)
num_observations = 5000

x1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_observations)
x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)

# generate simulated data
simulated_separableish_features = np.vstack((x1, x2)).astype(np.float32)
simulated_labels = np.hstack((np.zeros(num_observations),
                              np.ones(num_observations)))

np.random.seed(2)
num_observations = 5000

x3 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_observations)
x4 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)

# generate test data
t1 = np.vstack((x3, x4)).astype(np.float32)
t2 = np.hstack((np.zeros(num_observations),
                              np.ones(num_observations)))

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def loglikelihood(data, simulated_labels, weights):
    scores = np.matmul(weights, data)
    y_pred = sigmoid(scores)
    result = simulated_labels * np.log(y_pred) + (1-simulated_labels) * np.log(1-y_pred)
    return np.mean(result)

weights = np.zeros((1,3))
m = 10000
lr = 0.01
epoch = 10000
ones = np.ones((1,10000))
simulated_separableish_features = simulated_separableish_features.transpose()
data = np.concatenate((ones, simulated_separableish_features))
simulated_labels = np.expand_dims(simulated_labels, 0)

for step in range(epoch):
    scores = np.matmul(weights, data)
    y_pred = sigmoid(scores)
    delta = np.matmul((y_pred - simulated_labels), data.transpose()) / m
    weights = weights - lr * delta

    if step % 100 == 0:
        print(loglikelihood(data, simulated_labels, weights))

t1 = t1.transpose()
data = np.concatenate((ones, t1))
scores = np.matmul(weights, data)
y_pred = sigmoid(scores)

result = np.mean(np.abs((y_pred - t2)))
print result