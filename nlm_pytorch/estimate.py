import numpy as np


def estimate_noise(img):
    upper = img[:-2, 1:-1].flatten()
    lower = img[2:, 1:-1].flatten()
    left = img[1:-1, :-2].flatten()
    right = img[1:-1, 2:].flatten()
    central = img[1:-1, 1:-1].flatten()
    U = np.column_stack((upper, lower, left, right))
    c_estimated = np.dot(U, np.dot(np.linalg.pinv(U), central))
    error = np.mean((central - c_estimated)**2)
    sigma = np.sqrt(error)
    return sigma
