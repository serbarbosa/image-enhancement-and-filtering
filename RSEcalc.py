"""
author: Sergio Ricardo Gomes Barbosa Filho
nusp:   10408386
course: scc0251
year/semester: 2020/1
Assignment 2: image enhancement and filtering
"""
import math
import numpy as np

def get_rse(input_img, output_img):
    '''
        Performs the RSE calculation between the input_img and the output_img
        
        returns: float value of the calculated RSE
    '''
    #converting images values to float to avoid memory overflow
    input_img = input_img.astype(np.float32)
    output_img = output_img.astype(np.float32)
    
    rse = 0

    for i in range(len(input_img)):
        for j in range(len(input_img[0])):
            #adding the square of the subtraction of the corresponding pixels
            rse += (input_img[i][j] - output_img[i][j]) ** 2.0
    
    #at last, returns the squared root of the summation
    return math.sqrt(rse)
