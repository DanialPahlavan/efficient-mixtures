import matplotlib.pyplot as plt
import numpy as np
import matplotlib


# MNIST
S_4 = [10.14, 16.66, 22.93, 29.45]
S_3 = [10.15, 16.78, 22.98]
S_2 = [10.14, 16.50]
S_1 = [10.11]
N = [1, 2, 3, 4]

matplotlib.rcParams.update({'font.size': 15})
fig, ax = plt.subplots(layout='constrained')
w = 0.1
for i, (t, c) in enumerate(zip((S_1[0], S_2[0], S_3[0], S_4[0]), ('black', 'orange', 'green', 'red'))):
    rects = ax.bar(0 + w * i, t, w, color=c, label=f'$S = {i + 1}$')
for i, (t, c) in enumerate(zip((S_2[1], S_3[1], S_4[1]), ('orange', 'green', 'red'))):
    rects = ax.bar(1 + w * i, t, w, color=c)
for i, (t, c) in enumerate(zip((S_3[2], S_4[2]), ('green', 'red'))):
    rects = ax.bar(2 + w * i, t, w, color=c)
rects = ax.bar(3, S_4[3], w, color='red')
plt.grid(1)
plt.xticks(np.arange(4), N)
plt.xlabel("$N$")
plt.ylabel("seconds/epoch")
ax.legend(loc='upper left')
plt.show()

"""
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
"""