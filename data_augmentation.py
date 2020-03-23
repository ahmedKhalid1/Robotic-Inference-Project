import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import imageio
import glob

# ia.seed(1)

# Example batch of images.
# The array has shape (32, 64, 64, 3) and dtype uint8.
count = 0
for filename in glob.glob('G:\\paper\\*.png'):
    if count == 1:
        imgs = np.concatenate([lastImg, np.array([imageio.imread(filename)])]) # Making it 3 dimensional so that the appending happens along the first dimension
        count += 1
    if count == 0:
        lastImg = np.array([imageio.imread(filename)])
        count += 1
    else:
        imgs = np.concatenate([imgs, np.array([imageio.imread(filename)])])

print(imgs.shape)
    

seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    iaa.Flipud(0.5),
    iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.ContrastNormalization((0.9, 1.1)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.9, 1.1), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.KeepSizeByResize(iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        rotate=(-10, 10),
        shear=(-8, 8),
        mode="symmetric")
    )
    #translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
], random_order=True) # apply augmenters in random order

for j in range(4):
    images_aug = seq.augment_images(imgs)
    for i in range(images_aug.shape[0]):
        imageio.imwrite('G:\\final_dataset\\paper\\{}_{}.png'.format(j, i), images_aug[i])