import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters

def harris_corner_detector(image, sobel_size, gauss_size, k=0.05):
    # Compute derivatives
    Ix = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=sobel_size)
    Iy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=sobel_size)

    # Compute products of derivatives and blur them
    Ix2 = cv2.GaussianBlur(Ix**2, (gauss_size, gauss_size), 0)
    Iy2 = cv2.GaussianBlur(Iy**2, (gauss_size, gauss_size), 0)
    Ixy = cv2.GaussianBlur(Ix*Iy, (gauss_size, gauss_size), 0)

    # Harris corner response
    det = Ix2 * Iy2 - Ixy**2
    trace = Ix2 + Iy2
    harris_response = det - k * trace**2

    return harris_response

def find_max(image, size, threshold):
    data_max = filters.maximum_filter(image, size)
    maxima = (image == data_max)
    diff = image > threshold
    maxima[diff == 0] = 0
    return np.nonzero(maxima)

def plot_corners(image, corners):
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.plot(corners[1], corners[0], '*', color='r')  # x and y are swapped due to imshow's coordinate system
    plt.show()

# Load images
fontanna1 = cv2.imread('fontanna1.jpg', cv2.IMREAD_GRAYSCALE)
fontanna2 = cv2.imread('fontanna2.jpg', cv2.IMREAD_GRAYSCALE)
budynek1 = cv2.imread('budynek1.jpg', cv2.IMREAD_GRAYSCALE)
budynek2 = cv2.imread('budynek2.jpg', cv2.IMREAD_GRAYSCALE)

# Parameters
sobel_size = 15
gauss_size = 15
threshold = 0.01

# Harris corner detection and plotting
# fontanna1_corners = find_max(harris_corner_detector(fontanna1, sobel_size, gauss_size), sobel_size, threshold)
# fontanna2_corners = find_max(harris_corner_detector(fontanna2, sobel_size, gauss_size), sobel_size, threshold)
budynek1_corners = find_max(harris_corner_detector(budynek1, sobel_size, gauss_size), sobel_size, threshold)
budynek2_corners = find_max(harris_corner_detector(budynek2, sobel_size, gauss_size), sobel_size, threshold)

# plot_corners(fontanna1, fontanna1_corners)
# plot_corners(fontanna2, fontanna2_corners)
plot_corners(budynek1, budynek1_corners)
plot_corners(budynek2, budynek2_corners)
