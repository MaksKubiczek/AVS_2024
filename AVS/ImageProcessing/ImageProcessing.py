import os
import cv2
from utilities.FileManager import FileManager

'''
ImageProcessing class

Methods:

    > rgb2gray

    > resize_image

    > resizing_images

    > images_addition

    > images_substraction

    > images_multiplication

    > images_linear_combination

    > images_abs_diff
'''


class ImageProcessing:
    
    @staticmethod
    def rgb2gray(Image):
        return 0.299*Image[:,:,0] + 0.587*Image[:,:,1] + 0.114*Image[:,:,2]

    @staticmethod
    def resize_image(Image, scale):
        height, width = Image.shape[:2]
        New_Image = cv2.resize(Image,(int(scale*height), int(scale*width)))
        return New_Image 

    @staticmethod
    def resizing_images(Image1, Image2):
        height, width = min(Image1.shape[0], Image2.shape[0]), min(Image1.shape[1], Image2.shape[1])
        Image1 = cv2.resize(Image1, (width, height))
        Image2 = cv2.resize(Image2, (width, height))
        return Image1, Image2

    @staticmethod
    def images_addition(Image_path_1, Image_path_2):
        Image1 = FileManager.read_from_path_grey(Image_path_1)
        Image2 = FileManager.read_from_path_grey(Image_path_2)
        Image1, Image2 = ImageProcessing.resizing_images(Image1, Image2)
        result = cv2.add(Image1, Image2)
        return result

    @staticmethod
    def images_substraction(Image_path_1, Image_path_2):
        Image1 = FileManager.read_from_path_grey(Image_path_1)
        Image2 = FileManager.read_from_path_grey(Image_path_2)
        Image1, Image2 = ImageProcessing.resizing_images(Image1, Image2)
        result = cv2.subtract(Image1, Image2)
        return result

    @staticmethod
    def images_multiplication(Image_path_1, Image_path_2, scale):
        Image1 = FileManager.read_from_path_grey(Image_path_1)
        Image2 = FileManager.read_from_path_grey(Image_path_2)
        Image1, Image2 = ImageProcessing.resizing_images(Image1, Image2)
        result = cv2.multiply(Image1, Image2, scale)
        return result

    @staticmethod
    def images_linear_combination(Image_path_1, weight_1, Image_path_2, weight_2, gammma):
        Image1 = FileManager.read_from_path_grey(Image_path_1)
        Image2 = FileManager.read_from_path_grey(Image_path_2)
        Image1, Image2 = ImageProcessing.resizing_images(Image1, Image2)
        result = cv2.addWeighted(Image1, weight_1, Image2, weight_2, gammma)
        return result


    @staticmethod
    def images_abs_diff(Image_path_1, Image_path_2):
        Image1 = FileManager.read_from_path_grey(Image_path_1)
        Image2 = FileManager.read_from_path_grey(Image_path_2)
        Image1, Image2 = ImageProcessing.resizing_images(Image1, Image2)
        result = cv2.absdiff(Image1, Image2)
        return result

