import os
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


img_dir = '/home/users/u7244940/clip_project/fig/pe'
files = os.listdir(img_dir)

png_files = [f for f in files if f.endswith('.png')]


n = len(png_files)

cols = 3
rows = math.ceil(n / cols)


fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))

axs = axs.ravel()


for i, file in enumerate(png_files):
    img_path = os.path.join(img_dir, file)
    img = mpimg.imread(img_path)
    axs[i].imshow(img)
    axs[i].set_title(file[:-4], fontsize=10)
    axs[i].axis('off')


for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout()

plt.savefig('./fig/cm_pe.png')
plt.show()
