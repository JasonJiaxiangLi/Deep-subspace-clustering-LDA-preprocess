from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

split = 2
pgm_dir = "./CroppedYale/yaleB01/yaleB01_P00A-005E+10.pgm"
img = Image.open(pgm_dir)
# img.show()
data_mat = np.array(img)
print(np.array(np.shape(data_mat)))
# We now do pooling and reshape
shape = np.array(np.shape(data_mat)) / split
shape = [int(i) for i in shape]
new_data = np.zeros(shape)
for i in range(shape[0]):
    for j in range(shape[1]):
        new_data[i, j] = data_mat[split * i, split * j]

print(shape[0],shape[1],shape[0]*shape[1])
plt.subplot(121)
plt.imshow(data_mat)
plt.subplot(122)
plt.imshow(new_data)
plt.show()
