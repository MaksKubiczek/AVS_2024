from ImageProcessing.ImageOperation import ImageOperation
from utilities.FileManager import FileManager
from utilities.Histrogram import Histrogram
import matplotlib.pyplot as plt
import cv2
import numpy as np
from Labs.lab1 import Lab1
from Labs.lab2 import Lab2
from Labs.lab3 import Lab3
from Labs.lab4 import Lab4


def main() -> None:
    #     #Lab1.lab1()

    #     # lab2 = Lab2()
    #     #lab2.set_parameters()
    #     #lab2.set_step_value(2)
    #     # lab2.lab2()
    # Lab3.lab3()
    # Lab4.block_method('I.jpg', 'J.jpg')

    Lab4.multiscale_block_method("I.jpg", "J.jpg")
    
    

if __name__ == "__main__":
    main()

