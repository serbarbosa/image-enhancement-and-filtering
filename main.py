'''
author: Sergio Ricardo Gomes Barbosa Filho
nusp:   10408386
course: scc0251
year/semester: 2020/1
Assignment 2: Image enhancement and filtering
'''
import numpy as np
#import matplotlib.pyplot as plt
import imageio
import math



def get_rse(input_img, output_img):
    '''
        Performs the RSE calculation between the input_img and the output_img
        
        returns: float value of the calculated RSE
    '''
    #converting images values to float to avoid memory overflow
    rse = 0

    for i in range(1, len(input_img)-1):
        for j in range(1, len(input_img[0]) -1):
            #adding the square of the subtraction of the corresponding pixels
            rse += (input_img[i][j] - output_img[i][j]) ** 2.0
    
    #at last, returns the squared root of the summation
    return math.sqrt(rse)


def get_gaussian_val(x, sigma):
    """
    Calculates and returns the gaussian kernel value for the given parameters
    """
    return math.exp(-(x**2)/(2*(sigma**2))) / 2 * math.pi * (sigma**2)
    
def bilateral_filter(img, f_size, sig_s, sig_r):
    
    N, M = img.shape  #dimensions of the image
    #initializing output img
    output_img = np.zeros(img.shape)

    #initializing spatial component with zeros
    spatial_component = np.zeros( (f_size, f_size) )
    a = int((f_size-1)/2)
    b = int((f_size-1)/2)

    #computing the spatial gaussian component for each position of the filter
    for i in range(-a, a+1):
        for j in range(-a, a+1):
            spatial_component[i][j] = get_gaussian_val( math.sqrt(i++2 + j**2), sig_s)
    
    #applying the convolution for each pixel in the image
    for i in range(a, N-a):         #'a' rolls unable to compute
        for j in range(b, M-b):     #'b' colums unable to compute
            
            If = 0      #value of centered pixel
            Wp = 0

            neighborhood = img[i-a : i+a+1, j-b : j+b+1]  #region centered at i,j
            If = 0      #new value for the centered pixel

            #for each neighbor inside the window size
            for k in range(-a,a+1):
                for l in range(-b,b+1):
                    #compute value of range component multiplying by correspondent spatial component
                    wi = spatial_component[k][l] * get_gaussian_val(neighborhood[k][l] - img[i][j], sig_r)
                    #computing normalization factor (needs to be done before computing the final filter value)
                    Wp += wi
                    
                    #multiplying filter value by its intensity and summing in the total value
                    If += neighborhood[k][l] * wi
            #applying normalization and saving to output 
            output_img[i][j] = If/Wp

        
    return output_img


# ------------ main -------------

if __name__ == "__main__":
    
    #first parameters input   
    img_filename = str(input()).rstrip()
    method = int(input())
    save = int(input()) == 1
    
    input_img = imageio.imread(img_filename).astype(np.float32)    #reading image
    input_img = np.pad(input_img, ((1,1), (1,1)), 'constant')

    output_img = None   #initializing output variable
    
    if method == 1:
        filter_size = int(input())
        sig_s = float(input())
        sig_r = float(input())
        output_img = bilateral_filter(input_img, filter_size, sig_s, sig_r)

    if method == 2:
        
        c = float(input())
        kernel = int(input())

    if method == 3:
        sig_row = float(input())
        sig_col = float(input())
    
    #calculating and printing the RSE
    rse = round(get_rse(input_img, output_img), 4)
    print(rse)

    #saving the modified image when required
    if save:
        output_img = output_img.astype(np.uint8)
        imageio.imwrite('output_img.png', output_img)



