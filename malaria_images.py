from skimage.io import imread
from skimage.transform import resize
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import numpy.ma

def make_image_loader(size=64, preprocess=None):
    """
    Create a function to load and resize an image from a path, with an optional transformation
    (e.g., standardization, augmentation) before resizing.
    """
    if preprocess is None:
        preprocess = lambda x: x
    def load_image(path):
        image = imread(path)
        image = image.astype('float') / 255
        image = preprocess(image)
        image = resize(image, (size, size))
        return image
    return load_image

def extract_mask(image):
    mask = (image.sum(axis=2) == 0).astype('float')
    mask = np.expand_dims(mask, axis=2).repeat(3, axis=2)
    return mask

def standardize_image(image, mask=None):
    """
    Shift the image to zero mean and unit variance, ignoring the black border region.
    """
    if mask is None:
        mask = extract_mask(image)
    image = np.ma.masked_array(image, mask)
    image -= image.mean(keepdims=True)
    image /= image.std(keepdims=True) + 1e-6
    image = image.filled(0.0)
    return image

#
# Augmentation
#
# Keras's ImageDataGenerator is a good starting point, but it doesn't deal with a few issues:
#  * The black border region should be ignored for color transformations and standardizing the image
#  * For maximum fidelity, the image should be resized *after* it is rotated/translated
#  * The built-in training/validation split doesn't handle groups
#

generator = ImageDataGenerator(
    rotation_range=360,
    horizontal_flip=True,
    vertical_flip=True,
    #brightness_range=(1.0, 1.0),
    channel_shift_range=0.2,
    fill_mode='constant',
    cval=1,
)

def pad_square(image):
    dims = np.array(image.shape)[:2]
    even_pad = (dims.max() - dims) / 2
    lower_pad = np.floor(even_pad).astype('int')
    upper_pad = np.ceil(even_pad).astype('int')
    pad = list(zip(lower_pad, upper_pad)) + [(0, 0)]
    return np.pad(image, pad, 'constant')

def augment_masked_image(image, mask):
    transform = generator.get_random_transform(image.shape)
    image = generator.apply_transform(image, transform)
    
    transform.update(brightness=None, channel_shift_intensity=None)
    mask = generator.apply_transform(mask, transform)
    
    return image, mask

def augment_image(image):
    image = pad_square(image)
    mask = extract_mask(image)
    image, mask = augment_masked_image(image, mask)
    image = standardize_image(image, mask)
    return image