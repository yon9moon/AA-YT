import random
from typing import Any
import cv2
import torch
import numpy as np

class RandomTranslate:
    def __init__(self, h, w, probability):
        assert  1 >= h >= 0 and 1 >= w >= 0 and 0 <= probability <= 1
        self.h = h
        self.w = w
        self.probability = probability
    
    def __call__(self, img):
        if np.random.rand() > self.probability:
            return img
        h = self.h
        w = self.w
        if h == 0 and w == 0:
            return img

        C, H, W = img.shape
        # print(C,H,W)
        translations = [(round(random.uniform(-w*W, w*W)), round(random.uniform(-h*H, h*H))) for _ in range(C)]
        # translations = [(-w*W, -h*H) for _ in range(C)]
        # print(translations)

        translated_channels = []
        for c in range(C):
            x, y = translations[c]
            m = np.float32([[1, 0, x], [0, 1, y]])
            channel_translated = cv2.warpAffine(img[c], m, (W, H))
            translated_channels.append(channel_translated)
        translated_img = np.stack(translated_channels, axis=0)    
        return translated_img
    
def test_RandomTranslate():
    path = '/home/lhb/npy/ytnpy/10105-1/20220921-1/-1 -1/0.npy'
    img = np.load(path)
    img = img.astype('float32')
    print(img.shape)
    img = [img]
    img = np.array(img, dtype=float)
    print(img.shape)
    cv2.imwrite('before.png', img.transpose(1, 2, 0).astype(np.uint8))
    tsfm = RandomTranslate(0,0)
    translated_img = tsfm(img)
    print(translated_img.shape)
    cv2.imwrite('after.png', translated_img.transpose(1, 2, 0).astype(np.uint8))

def test_RandomTranslate_2():
    path = '/home/lhb/npy/ytnpy/10105-1/20220921-1/-1 -1/0.npy'
    path1 = '/home/lhb/npy/ytnpy/10105-1/20220921-1/-1 -1/1.npy'
    img0 = np.load(path)
    img0 = img0.astype('float32')
    print(img0.shape)
    img1 = np.load(path1)
    img1 = img1.astype('float32')
    img = [img0,img1]

    img = np.array(img, dtype=float)
    print(img.shape)
    cv2.imwrite('before0.png', img[0:1,:,:].transpose(1, 2, 0).astype(np.uint8))
    cv2.imwrite('before1.png', img[1:2,:,:].transpose(1, 2, 0).astype(np.uint8))
    tsfm = RandomTranslate(100,100)
    translated_img = tsfm(img)
    print(translated_img.shape)
    cv2.imwrite('after0.png', translated_img[0:1,:,:].transpose(1, 2, 0).astype(np.uint8))
    cv2.imwrite('after1.png', translated_img[1:2,:,:].transpose(1, 2, 0).astype(np.uint8))
