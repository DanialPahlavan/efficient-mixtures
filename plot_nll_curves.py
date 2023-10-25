import matplotlib.pyplot as plt
import numpy as np
import matplotlib


# MNIST
S_4 = [79.72, 78.73, 78.572, 78.52]
S_3 = [79.69, 78.81, 78.76]
S_2 = [79.75, 79.07]
S_1 = [79.81]
N = [1, 2, 3, 4]

matplotlib.rcParams.update({'font.size': 15})
fig, ax = plt.subplots(layout='constrained')
# plt.title(r'Trained with the some-to-some estimator')
plt.plot(S_1, label='$S=1$', marker="o", ls='None', markersize=10, markeredgecolor="red", markerfacecolor="black")
plt.plot(S_2, label='$S=2$')
plt.plot(S_3, label='$S=3$')
plt.plot(S_4, label='$S=4$')
plt.xticks(np.arange(4), N)
plt.xlabel("$N$")
plt.ylabel("NLL")
plt.legend()
plt.grid(1)
#plt.tight_layout()
plt.show()

# CIFAR
S_1 = [3.437]
S_2 = [3.2410]
S_3 = [3.2402, 3.2330]
S_4 = [3.2312, 3.2289, 3.2275]

matplotlib.rcParams.update({'font.size': 15})
plt.title(r'Trained with the some-to-some estimator')
plt.plot(0, S_1, label='$S=1$', marker="o", ls='None', markersize=15, markeredgecolor="red", markerfacecolor="black")
plt.plot(1, S_2, label='$S=2$', marker=10, ls='None', markersize=15)
plt.plot([1, 2], S_3, label='$S=3$', marker='p', ls='None', markersize=15, alpha=0.5)
plt.plot([1, 2, 3], S_4, label='$S=4$', marker="*", ls='None', markersize=15)
plt.xlabel("$N$")
plt.ylabel("BPD")
plt.xticks(np.arange(4), N)
plt.legend()
plt.grid(1)
plt.tight_layout()
plt.show()
