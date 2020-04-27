import numpy as np
a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

c = np.reshape(a, (-1, 4))

print(c)

print(np.delete(c, 0, 1))
