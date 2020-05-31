import numpy as np

print(np.array([1, 2, 3]))

print(np.arange(1, 10, 2))

print(np.zeros((2, 3)))
print(np.ones((2, 3)))

print(np.linspace(1, 100, 10))

print(np.random.rand(2, 3))
print(np.random.randn(2, 3))

arr = np.arange(1, 26)
arr = arr.reshape(5, 5)
print(arr)

print('head')
print(arr[:2])

print('tail')
print(arr[-2:])

print('left')
print(arr[:,:-2])

print('right')
print(arr[:,-2:])

print('+++')
def compute_labels(set):
    sums = np.sum(set, axis=0)
    labels = list(map(lambda x: int(x*100)%2, sums))
    return np.array(labels).reshape(1, set.shape[1])

data = np.random.randn(5, 5)
print(data)
print(compute_labels(data))


# train = arr
# labels = np.square(np.sum(train, axis=0))
# print(train)
# print(labels)
# print(output(train))
