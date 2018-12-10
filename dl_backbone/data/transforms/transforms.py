import random, math
import numpy as np
import cv2
import torch.nn.functional as F


class Resize(object):
    """
    Resize given image
    Args:
        size (H, W)
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, bgry):
        """
        Args:
            bgry: (H, W, BGRY)

        Returns:
            np.array: numpy.ndarray (H, W, BGRY)
        """
        height, width, _ = bgry.shape
        if (height, width) != self.size:
            bgr_img = bgry[:, :, [0, 1, 2]]
            y_channel = bgry[:, :, 3]
            bgr_img = cv2.resize(bgr_img, self.size)
            y_channel = cv2.resize(y_channel, self.size)

            y_img = np.expand_dims(y_channel, axis=2)
            bgry = np.concatenate((bgr_img, y_img), axis=2)
        return bgry

    def __repr__(self):
        return self.__class__.__name__ + '(size={})'.format(self.size)


class RandomHVFlip(object):
    """Horizontally/Vertically flip the given Image
    randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
        mode (int): 1 for vertical, 0 for horizontal
    """

    def __init__(self, mode, p=0.5):
        self.mode = mode
        self.p = p

    def __call__(self, bgry):
        """
        Args:
            bgry: (H, W, BGRY)

        Returns:
            np.array: numpy.ndarray (H, W, BGRY)
        """
        if random.random() < self.p:
            bgr_img = bgry[:, :, [0, 1, 2]]
            y_channel = bgry[:, :, 3]
            bgr_img = cv2.flip(bgr_img, self.mode)
            y_channel = cv2.flip(y_channel, self.mode)

            y_img = np.expand_dims(y_channel, axis=2)
            bgry = np.concatenate((bgr_img, y_img), axis=2)
        # assert np_bgry.shape[2] == 4
        return bgry

    def __repr__(self):
        return self.__class__.__name__ + '(p={}, mode={})'.format(self.p, self.mode)


class RandomClockwiseRotate(object):
    """clockwise/counterclockwise rotate the given Image
    randomly with a given probability.

        Args:
            p (float): probability of the image being flipped. Default value is 0.5
            mode (int): 1 for cw, 0 for ccw
    """
    def __init__(self, mode, p=0.5):
        self.mode = mode
        self.p = p

    def __call__(self, bgry):
        """
        Args:
            bgry: (H, W, BGRY)

        Returns:
            np.array: numpy.ndarray (H, W, BGRY)
        """
        if random.random() < self.p:
            bgr_img = bgry[:, :, [0, 1, 2]]
            y_channel = bgry[:, :, 3]
            bgr_img = cv2.transpose(bgr_img)
            bgr_img = cv2.flip(bgr_img, flipCode=self.mode)

            y_channel = cv2.transpose(y_channel)
            y_channel = cv2.flip(y_channel, flipCode=self.mode)

            y_img = np.expand_dims(y_channel, axis=2)
            bgry = np.concatenate((bgr_img, y_img), axis=2)
        # assert np_bgry.shape[2] == 4
        return bgry

    def __repr__(self):
        return self.__class__.__name__ + '(p={}, mode={})'.format(self.p, self.mode)


class RandomRotation(object):
    """
    Rotate given image
    Args:
        degree (start, end)
    """
    def __init__(self, degree, is_train=True):
        self.degree = degree
        self.is_train = is_train

    def __call__(self, bgry):
        """
        Args:
            bgry: (H, W, BGRY)

        Returns:
            np.array: numpy.ndarray (H, W, BGRY)
        """
        if self.is_train:
            h, w, _ = bgry.shape
            center = (int(w / 2), int(h / 2))
            degree = random.randint(*self.degree)
            M = cv2.getRotationMatrix2D(center, degree, 1.0)

            bgr_img = bgry[:, :, [0, 1, 2]]
            y_channel = bgry[:, :, 3]
            bgr_img = cv2.warpAffine(bgr_img, M, (w, h))
            y_channel = cv2.warpAffine(y_channel, M, (w, h))

            y_img = np.expand_dims(y_channel, axis=2)
            bgry = np.concatenate((bgr_img, y_img), axis=2)
        return bgry

    def __repr__(self):
        return self.__class__.__name__ + '(degree={}, is_train={})'.format(self.degree, self.is_train)
