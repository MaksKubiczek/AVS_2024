import os
import cv2
import numpy as np
from utilities.FileManager import FileManager

'''
imageProcessing class

Methods:

    > rgb2gray(image)

    > resize_image(image, scale)

    > resizing_images(image1, image2)

    > images_addition(image_path_1, image_path_2)

    > images_substraction(image_path_1, image_path_2)

    > images_multiplication(image_path_1, image_path_2, scale)

    > images_linear_combination(image_path_1, weight_1, image_path_2, weight_2, gammma)

    > images_abs_diff(image_path_1, image_path_2)

    > binarization(threshold, difference)
'''


class ImageOperation:
    
    @staticmethod
    def rgb2gray(image):
        return 0.299*image[:,:,0] + 0.587*image[:,:,1] + 0.114*image[:,:,2]

    @staticmethod
    def resize_image(image, scale):
        height, width = image.shape[:2]
        New_image = cv2.resize(image,(int(scale*height), int(scale*width)))
        return New_image 

    @staticmethod
    def resizing_images(image1, image2):
        height, width = min(image1.shape[0], image2.shape[0]), min(image1.shape[1], image2.shape[1])
        image1 = cv2.resize(image1, (width, height))
        image2 = cv2.resize(image2, (width, height))
        return image1, image2

    @staticmethod
    def images_addition(image_path_1, image_path_2):
        image1 = FileManager.read_from_path_grey(image_path_1)
        image2 = FileManager.read_from_path_grey(image_path_2)
        image1, image2 = ImageProcessing.resizing_images(image1, image2)
        result = cv2.add(image1, image2)
        return result

    @staticmethod
    def images_substraction(image_path_1, image_path_2):
        image1 = FileManager.read_from_path_grey(image_path_1)
        image2 = FileManager.read_from_path_grey(image_path_2)
        image1, image2 = ImageProcessing.resizing_images(image1, image2)
        result = cv2.subtract(image1, image2)
        return result

    @staticmethod
    def images_multiplication(image_path_1, image_path_2, scale):
        image1 = FileManager.read_from_path_grey(image_path_1)
        image2 = FileManager.read_from_path_grey(image_path_2)
        image1, image2 = ImageProcessing.resizing_images(image1, image2)
        
        # Konwersja na float32
        image1_float = image1.astype(np.float32)
        image2_float = image2.astype(np.float32)
        
        # Mnożenie obrazów
        result_float = cv2.multiply(image1_float, image2_float, scale)
        #result_float = image1_float*image2_float
        
        return result_float

    @staticmethod
    def images_linear_combination(image_path_1, weight_1, image_path_2, weight_2, gammma):
        image1 = FileManager.read_from_path_grey(image_path_1)
        image2 = FileManager.read_from_path_grey(image_path_2)
        image1, image2 = ImageProcessing.resizing_images(image1, image2)
        result = cv2.addWeighted(image1, weight_1, image2, weight_2, gammma)
        return result


    @staticmethod
    def images_abs_diff(image_path_1, image_path_2):
        image1 = FileManager.read_from_path_grey(image_path_1)
        image2 = FileManager.read_from_path_grey(image_path_2)
        image1, image2 = ImageProcessing.resizing_images(image1, image2)
        result = cv2.absdiff(image1, image2)
        return result

    @staticmethod
    def binarization(threshold, difference):
        return 1*(difference > threshold)

    @staticmethod
    def erosion(image, kernel, iterations):
        return cv2.erode(image, kernel, iterations)


    @staticmethod
    def dilation(image, kernel, iterations):
        return cv2.dilate(image, kernel, iterations)
    
                # # Perform erosion and dilation
                # kernel = np.ones((5, 5), np.uint8)
                # erosion = cv2.erode(binary_result, kernel, iterations=1)
                # dilation = cv2.dilate(erosion, kernel, iterations=1)