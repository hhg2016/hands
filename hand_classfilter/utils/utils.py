from PIL import Image
import cv2
import numpy

# img = cv2.imread('1.jpg')
#
# print(img[:,:,0][130,130]) #b
#
# image_open = Image.open('1.jpg')
# r, g, b = image_open.split()
# print(b.getpixel((130, 130)))#b
#
# im = numpy.array(img)  # 转换为numpy
# fromarray = Image.fromarray(img.astype('uint8')).convert('RGB')
# r, g, b = fromarray.split()
# print(b.getpixel((130, 130)))
#
# color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# print(color[...,2][130][130])#b
#
# color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
# print(color[...,2][130][130])#b

# import os
#
# listdir = os.listdir('dataset/hands6/test')
# for d in listdir:
#     img_list = os.listdir(os.path.join('dataset/hands6/test', d))
#     for idx, img in enumerate(img_list):
#         os.rename(f'dataset/hands6/test/{d}/{img}', f'dataset/hands6/test/{d}/test_{idx}.bmp')


x=[10]
xp = (100,150)
fp=(200,300)
y = numpy.interp(x, xp, fp)
print(y)