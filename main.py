'''
author: Sergio Ricardo Gomes Barbosa Filho
nusp:   10408386
course: scc0251
year/semester: 2020/1
Assignment 2: Image enhancement and filtering
'''
import numpy as np
import matplotlib.pyplot as plt
import imageio







# ------------ main -------------

if __name__ == "__main__":
    
    #first parameters input   
    img_filename = str(input()).rstrip()
    method = int(input())
    save = int(input()) == 1
    
    input_img = imageio.imread(img_filename)    #reading image
    output_img = None   #initializing output variable
    
    if method == 1:
        filter_size = int(input())
        sig_s = float(input())
        sig_r = float(input())

    if method == 2:
        
        c = float(input())
        kernel = int(input())

    if method == 3:
        sig_row = float(input())
        sig_col = float(input())
    
    #saving the modified image when required
    if save:
        imageio.imwrite('output_img.png', output_img)
    
    #calculating and printing the RSE
    rse = round(get_rse(input_img, output_img), 4)
    print(rse)




