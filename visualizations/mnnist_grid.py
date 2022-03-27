import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

PATH = 'C:\\Users\\Raz\\Desktop\\Studies\\PhD\\Adversarial Attacks\\MyPaper\\images\\CIFAR10\\'

def img_reshape(path, img):
    img = Image.open(path+'\\'+img).convert('RGB')
    img = img.resize((300,300))
    img = np.asarray(img)
    return img

main_dir = os.listdir(PATH)
img_arr = []

for dir in main_dir:
    path_cur_dir = os.path.join(PATH, dir)
    cur_dir = os.listdir(path_cur_dir)
    for image in cur_dir:
        img_arr.append(img_reshape(path_cur_dir, image))


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

# plt.axis('off')




fig = plt.figure(figsize=(20., 20.))

grid = ImageGrid(fig, 111,
                 nrows_ncols=(10, 10),  # creates 2x2 grid of axes
                 axes_pad=0.2,  # pad between axes
                 share_all=True)

grid[0].get_yaxis().set_ticks([])
grid[0].get_xaxis().set_ticks([])
#
for i, (ax, im) in enumerate(zip(grid, img_arr)):
    ax.imshow(im)
#     ax.set_ylabel(i // 10, rotation=90, fontsize=30)
#
# for i, ax in enumerate(grid):
#     ax.tick_params(top=True, bottom=False)
#     ax.set_xlabel(i % 10, fontsize=30, loc='center')

plt.show()

fig.savefig('CIFAR10.jpg', dpi=fig.dpi)
