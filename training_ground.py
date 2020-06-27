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
print(arr[:, :-2])

print('right')
print(arr[:, -2:])

print('+++')


def compute_labels(set):
    sums = np.sum(set, axis=0)
    labels = list(map(lambda x: int(x * 100) % 2, sums))
    return np.array(labels).reshape(1, set.shape[1])


arr = np.random.rand(5, 5)
print(arr)
print("sum")
print(np.sum(arr, axis=0))
print("* 100")
print(np.sum(arr, axis=0) * 100)
print("even or odd")
print(compute_labels(arr))

a = np.power(np.random.randn(1, 4), 2)
print(a)
b = np.sin(a)
b = b > 0.5
print(b)

# ---

x = np.arange(100).reshape(10, 10)
y = np.arange(10).reshape(1, 10)
print(x)
# print(y)
p = list(np.random.permutation(10))
print(p)

print(x[p, :])
# print(y[p, :])
