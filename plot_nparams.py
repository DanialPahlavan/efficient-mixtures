import matplotlib.pyplot as plt
import matplotlib
import numpy as np


# MNIST
S_4 = [607741]
S_3 = [607501]
S_2 = [607261]
S_1 = [607021]

matplotlib.rcParams.update({'font.size': 15})
fig, ax = plt.subplots(layout='constrained')
w = 1
for i, (t, c) in enumerate(zip((S_1[0], S_2[0], S_3[0], S_4[0]), ('black', 'orange', 'green', 'red'))):
    rects = ax.bar(i, t, w, color=c, label=f'$S = {i + 1}$')
    ax.bar_label(rects, padding=3)
# plt.grid(1)
plt.xticks(np.arange(4), [1, 2, 3, 4])
plt.xlabel("$S$")
plt.ylabel("#network parameters")
plt.ylim(606000, 607900)
plt.yticks([])
# plt.legend(loc='upper left', ncol=4)
plt.show()
