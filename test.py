import numpy as np


def multiply_through(index, list):
    if index == 0:
        return list[0]

    curr = list[0]
    i = 1
    while i <= index:
        print(curr)
        curr = curr @ list[i]
        i += 1
    return curr

matrices = []

a = np.array([[1,2,3],[4,5,6],[7,8,9]])
b = np.array([[1,0,0],[4,1,6],[2,3,0]])
c = np.array([[9,11,12],[4,7,6],[3,2,0]])
d = np.array([[1,2,3],[4,5,6],[7,8,9]])

matrices.append(a)
matrices.append(b)
matrices.append(c)
matrices.append(d)

result = multiply_through(2, matrices)
print("Result")
print(result)

print(a @ b)
print(a @ b @ c)