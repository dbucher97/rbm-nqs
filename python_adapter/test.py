import rbm
import numpy as np
import matplotlib.pyplot as plt

ham = rbm.get_hamiltonian(3)
lr = 0.01
# mat = expm(lr * ham.tocsc())
e, psi = rbm.groundstate(ham)
print("Groundstate found")

def normalized(v : np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)

def step(v : np.ndarray) -> np.ndarray:
    v -= lr * ham @ v + lr * np.random.normal(0, 0.001, v.shape[0])
    return normalized(v)

v = np.random.normal(1, 0.01, ham.shape[0]) + 1j * np.random.normal(0, 0.01, ham.shape[0])
v = normalized(v)

x = np.arange(1000)
y = []
for i in x:
    print(i)
    v = step(v)
    y.append(rbm.evaluate(v, ham).real / 8 - e / 8)

plt.plot(x, y)
plt.yscale('log')
plt.show()
