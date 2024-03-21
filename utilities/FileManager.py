import os
import cv2

'''
FileManager class

Methods:

    > ensure_folder_exists

    > save_to_png

    > save_to_jpg

    > save_to_png_path

    > save_to_jpg_path

    > read_from_png

    > read_from_jpg

    > read_from_path

    > read_from_path_grey

'''

class FileManager:
    @staticmethod
    def ensure_folder_exists(folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    @staticmethod
    def save_to_png(filename, image_to_save):
        cv2.imwrite(f"resources\pictures/modified/{filename}.png", image_to_save)

    @staticmethod
    def save_to_jpg(filename, image_to_save):
        cv2.imwrite(f"resources\pictures/modified/{filename}.jpg", image_to_save)

    @staticmethod
    def save_to_png_path(folder_path, file_name, image_to_save):
        folder_path = os.path.join("resources\pictures", folder_path)
        FileManager.ensure_folder_exists(folder_path)
        file_path = os.path.join(folder_path, f"{file_name}.png")
        cv2.imwrite(file_path, image_to_save)

    @staticmethod
    def save_to_jpg_path(folder_path, file_name, image_to_save):
        folder_path = os.path.join("resources\pictures", folder_path)
        FileManager.ensure_folder_exists(folder_path)
        file_path = os.path.join(folder_path, f"{file_name}.jpg")
        cv2.imwrite(file_path, image_to_save)

    @staticmethod
    def read_from_png(folder_path, filename):
        path = f"{folder_path}/{filename}.png"
        image = cv2.imread(path)
        if image is None:
            print(f"Error: Unable to read the image at {path}")
        return image

    @staticmethod
    def read_from_jpg(folder_path, filename):
        path = f"{folder_path}/{filename}.jpg"
        image = cv2.imread(path)
        if image is None:
            print(f"Error: Unable to read the image at {path}")
        return image

    @staticmethod
    def read_from_path(file_path):
        image = cv2.imread(file_path)
        if image is None:
            print(f"Error: Unable to read the image at {file_path}")
        return image

    @staticmethod
    def read_from_path_grey(file_path):
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Error: Unable to read the image at {file_path}")
        return image
