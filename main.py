'''
author: Sergio Ricardo Gomes Barbosa Filho

nusp:   10408386

course: scc0251

year/semester: 2020/1

git repository: github.com/serbarbosa/image-enhancement-and-filtering

Assignment 2: Image enhancement and filtering
'''

import numpy as np
#import matplotlib.pyplot as plt
import imageio
import math


# ----- helper functions ------
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

def scale_image(img):
    '''
        Scales the image using normalization(0 - 255) accordingly to the equation
        given by the pdf
    '''
    min_val = img.min()
    max_val = img.max()

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j] = (img[i][j] - min_val)*255/(max_val-min_val)

def get_gaussian_val(x, sigma):
    """
    Calculates and returns the gaussian kernel value for the given parameters
    """
    return math.exp(-(x**2)/(2*(sigma**2))) / 2 * math.pi * (sigma**2)


# ----- Methods implementation ------

def vignette_filter(input_img, sig_row, sig_col):
    ''' Performs third method '''
    
    rows, cols = input_img.shape  #dimensions of the image
    # initializing w_row and w_col as empty lists
    w_row = []
    w_col = []

    # calculating range to use as reference for gaussian
    row_start, row_end = int(-rows/2), int(rows/2)
    if rows % 2 == 0:   #even number of rows
        row_start = -int(rows/2 - 1)
    col_start, col_end = int(-cols/2), int(cols/2)
    if cols % 2 == 0:   #even number of columns
        col_start = -int(cols/2 - 1)

    # calculating 1D gaussian for w_row
    for i in range(row_start, row_end+1):
        w_row.append(get_gaussian_val(i, sig_row))
    
    # calculating 1D gaussian for w_col
    for i in range(col_start, col_end+1):
        w_col.append(get_gaussian_val(i, sig_col))
    
    # turning lists into numpy arrays
    w_row = np.array(w_row)
    w_col = np.array(w_col)
    
    #multiplying w_col transposed by w_row
    #-> there has been a confusion of which shape order to consider
    # for rows and columns. I chose the configuration that worked
    w_res = np.matmul(w_row[:, None], w_col[None, :])
    
    # multiplying the original image by the 2D kernel
    # and normalizing
    output_img = np.multiply(w_res, input_img)
    scale_image(output_img)

    return output_img

def unsharp_mask(input_img, c, kernel_op):
    ''' Performs second method '''

    #defining which kernel to use
    kernel =  np.matrix([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    if kernel_op == 2:
        kernel = np.matrix([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    
    N, M = input_img.shape  #image dimensions 
    n, m = kernel.shape     #kernel dimensions
    a = int((n-1)/2)
    b = int((m-1)/2)
    
    #initializing output img
    output_img = np.zeros(input_img.shape, dtype=np.float32)
    
    #will compute for each pixel reachable
    for i in range(a, N-a):
        for j in range(b, M-b):
            #selecting centered region
            neighborhood = input_img[i-a : i+a+1, j-b : j+b+1]
            #applying filter
            output_img[i, j] = np.sum( np.multiply(neighborhood, kernel))
    
    #performing first scaling operation
    scale_image(output_img)
    #performing addition
    output_img = output_img * c + input_img
    #performing second scaling operation
    scale_image(output_img)
    
    return output_img

    
def bilateral_filter(input_img, f_size, sig_s, sig_r):
    ''' Performs first method '''
    
    N, M = input_img.shape  #dimensions of the image
    #initializing output img
    output_img = np.zeros(input_img.shape)

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

            neighborhood = input_img[i-a : i+a+1, j-b : j+b+1]  #region centered at i,j
            If = 0      #new value for the centered pixel

            #for each neighbor inside the window size
            for k in range(-a,a+1):
                for l in range(-b,b+1):
                    #compute value of range component multiplying by correspondent spatial component
                    wi = spatial_component[k][l] * get_gaussian_val(neighborhood[k][l] - input_img[i][j], sig_r)
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
    input_img = np.pad(input_img, ((1,1), (1,1)), 'constant')       #adds padding(border of zeros)

    output_img = None   #initializing output variable
    
    if method == 1:
        
        filter_size = int(input())
        sig_s = float(input())
        sig_r = float(input())
        output_img = bilateral_filter(input_img, filter_size, sig_s, sig_r)

    if method == 2:
        
        c = float(input())
        kernel_op = int(input())
        output_img = unsharp_mask(input_img, c, kernel_op)

    if method == 3:
        
        sig_row = float(input())
        sig_col = float(input())
        output_img = vignette_filter(input_img, sig_row, sig_col)

    #calculating and printing the RSE
    rse = round(get_rse(input_img, output_img), 4)
    print(rse)

    #saving the modified image when required
    if save:
        output_img = output_img.astype(np.uint8)
        imageio.imwrite('output_img.png', output_img)



