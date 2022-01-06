import cv2
import numpy as np
import torch.nn as nn
from PIL import Image
from skimage import img_as_ubyte, color
from torchvision import models, transforms


def generate_feature_sift(image):
    r"""Extract scale invariant feature transform (SIFT) features from a single image.

    Parameters
    ----------
    image : ndarray
        Image data.

    Returns
    -------
    xfeat : ndarray
        Array of features. Has shape :math:`(N, 128)`, where :math:`N` is the number of features extracted.
    """

    sift = cv2.SIFT_create()
    _, xfeat = sift.detectAndCompute(image, None)

    return xfeat


def generate_feature_surf(image):
    r"""Extract speeded-up robust features (SURF) from a single image.

    Parameters
    ----------
    image : ndarray
        Image data.

    Returns
    -------
    xfeat : ndarray
        Array of features. Has shape :math:`(N, 64)`, where :math:`N` is the number of features extracted.
    """

    surf = cv2.xfeatures2d.SURF_create()
    _, xfeat = surf.detectAndCompute(image, None)

    return xfeat


def generate_feature_cnn_flatten(image, cnn='alexnet'):
    r"""Generate feature vector from final convolution layer by flattening.

    Parameters
    ----------
    image : ndarray
        Image data.
    cnn : str, optional
        'alexnet' (default)
            Generates CNN features from AlexNet architecture.
        'vgg'
            Generates CNN features from VGG architecture.

    Returns
    -------
    xfeat : ndarray
        Array of features. Has shape :math:`(N, )`, where :math:`N` is the number of voxels in the final convolution
        output.
    """

    if cnn == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif cnn == 'vgg':
        model = models.vgg19(pretrained=True)
    else:
        print('cnn must be either `alexnet` or `vgg`')
        return 1

    new_classifier = nn.Sequential(*list(model.features.children())[:-1])
    model = new_classifier

    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(Image.fromarray(color.gray2rgb(img_as_ubyte(image / np.max(image)))))
    input_batch = input_tensor.unsqueeze(0)
    output = model(input_batch).detach().numpy()
    xfeat = output.flatten()

    return xfeat


def generate_feature_cnn_maxpool(image, cnn='alexnet'):
    r"""Generate single feature vector from final convolution layer of CNN via apply MaxPooling.

    Parameters
    ----------
    image : ndarray
        Image data.
    cnn : str, optional
        'alexnet' (default)
            Generates CNN features from AlexNet architecture.
        'vgg'
            Generates CNN features from VGG architecture.

    Returns
    -------
    xfeat : ndarray
        Array of features. Has shape :math:`(d, )`, where :math:`d` is length of each feature output from the final
        convolution layer.
    """

    if cnn == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif cnn == 'vgg':
        model = models.vgg19(pretrained=True)
    else:
        print('cnn must be either `alexnet` or `vgg`')
        return 1

    new_classifier = nn.Sequential(*list(model.features.children())[:-1])
    model = new_classifier

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(Image.fromarray(color.gray2rgb(img_as_ubyte(image / np.max(image)))))
    input_batch = input_tensor.unsqueeze(0)
    output = model(input_batch).detach().numpy()
    xfeat = np.amax(output, axis=(2, 3))[0, :]

    return xfeat


def generate_feature_cnn_featdict(image, cnn='alexnet'):
    r"""Generate dictionary of features from CNN output.

    Parameters
    ----------
    image : ndarray
        Image data.
    cnn : str, optional
        'alexnet' (default)
            Generates CNN features from AlexNet architecture.
        'vgg'
            Generates CNN features from VGG architecture.

    Returns
    -------
    xfeat : ndarray
        Array of features. Has shape :math:`(N, d)`, where :math:`N` is the number of features in the final convolution
        output and d is the dimension of each feature.
    """

    if cnn == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif cnn == 'vgg':
        model = models.vgg19(pretrained=True)
    else:
        print('cnn must be either `alexnet` or `vgg`')
        return 1

    new_classifier = nn.Sequential(*list(model.features.children())[:-1])
    model = new_classifier

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(Image.fromarray(color.gray2rgb(img_as_ubyte(image / np.max(image)))))
    input_batch = input_tensor.unsqueeze(0)
    output = model(input_batch).detach().numpy()
    xfeat = np.reshape(output, (output.shape[1], -1))
    xfeat = np.transpose(xfeat)

    return xfeat
