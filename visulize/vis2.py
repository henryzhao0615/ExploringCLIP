import numpy as np
import matplotlib.pyplot as plt
import os


conf_matrix = np.array([
    [971,   2,   4,   3,   0,   0,   0,   0,  14,   6],
    [  4, 973,   0,   0,   0,   2,   1,   0,   4,  16],
    [ 13,   0, 917,  17,  29,   7,  14,   3,   0,   0],
    [  0,   0,   6, 910,  10,  53,  14,   4,   2,   1],
    [  3,   1,  12,  12, 937,   9,  13,  12,   0,   1],
    [  5,   1,   6,  60,  11, 910,   4,   3,   0,   0],
    [  5,   1,  15,  26,  15,   4, 934,   0,   0,   0],
    [  3,   1,   2,   9,  19,  10,   0, 956,   0,   0],
    [ 22,   1,   1,   0,   0,   0,   0,   1, 971,   4],
    [  2,  19,   0,   1,   0,   0,   0,   1,   5, 972]
])


classnames = ['airplane', 'automobile', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck']


plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()


tick_marks = np.arange(len(classnames))
plt.xticks(tick_marks, classnames, rotation=45)
plt.yticks(tick_marks, classnames)

thresh = conf_matrix.max() / 2.0
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()

if not os.path.exists("fig"):
    os.makedirs("fig")
save_path = os.path.join("fig", "confusion_matrix_full_shot.png")
plt.savefig(save_path)
plt.show()
