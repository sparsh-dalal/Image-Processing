import cv2
import numpy as np
from collections import deque

# For Binary Image
def BinaryImage_inRange(HSV_image):
    height, width = HSV_image.shape[:2]
    mask = np.zeros((height,width),np.uint8)

    for i in range(height):
        print(f"wait.{380 + 380 + 380 + height-i}")
        for j in range (width):

                h,s,v = HSV_image[i,j]
                if(h in range(0,28) and s in range(60,75) and v in range(0,200)):
                    mask[i,j] = 255

    return mask

# Function for zero Padding
def ZeroPadding(img,pad_size):
    height,width = img.shape
    new_height = height + 2*pad_size
    new_width = width + 2*pad_size

    img_out = np.zeros((new_height,new_width),np.uint8)

    img_out[pad_size:-pad_size,pad_size:-pad_size] = img

    return img_out

# Check whether kernel is a hit or not
def isHit(arr,kernel):
    h,w = kernel.shape

    for i in range(h):
        for j in range(w):
            if (arr[i,j]==255 and kernel[i,j]==255):
                return True
            
    return False

# Check whether Kernel is Fit or not
def isFit(arr,kernel):
    h,w = kernel.shape

    cnt = 0
    for i in range(h):
        for j in range(w):
            if (arr[i,j]==kernel[i,j]):
                cnt = cnt + 1

    if (cnt==h*w):return True
    return False
            
# Function for dilation 
def dilation(img,kernel):
    height,width = img.shape

    kh,kw = kernel.shape
    pad_size = int(kh/2)

    padded_image = ZeroPadding(img,pad_size)

    new_img = np.zeros((height,width),np.uint8)

    for i in range(height):
        #print(f"wait.{380 + height-i}")
        for j in range(width):

            # x and y are center of the mask Overlapping over the padded image
            x = i + pad_size
            y = j + pad_size

            arr = padded_image[x-pad_size:x+pad_size+1,y-pad_size:y+pad_size+1]

            if isHit(arr,kernel):
                new_img[i,j] = 255

    return new_img

# Function for erosion
def erosion(img,kernel):
    height,width = img.shape

    kh,kw = kernel.shape
    pad_size = int(kh/2)

    padded_image = ZeroPadding(img,pad_size)

    new_img = np.zeros((height,width),np.uint8)

    for i in range(height):
        print(f"wait.{380 + height-i}")
        for j in range(width):

            # x and y are center of the mask Overlapping over the padded image
            x = i + pad_size
            y = j + pad_size

            arr = padded_image[x-pad_size:x+pad_size+1,y-pad_size:y+pad_size+1]

            if isFit(arr,kernel):
                new_img[i,j] = 255

    return new_img

# Function for closing , dilation followed by erosion
def Closing(img,kernel):

    img1 = dilation(img,kernel)
    img2 = erosion(img1,kernel)

    return img2

# Function for Opening , ersosion followed by Dilation
def Opening(img,kernel):

    img1 = erosion(img,kernel)
    img2 = dilation(img1,kernel)

    return img2

# Finding the longest connected Component in the image
def longest_connected_component(image):
    # Going to use queue data structure

    height,width = image.shape

    longest_component = []
    vis = np.zeros((height,width),np.uint8)
    # Making the vis array to mark and check whether a point is visited or not

    for i in range(height):
        print(f"wait.{height-i}")
        for j in range(width):
            if (image[i,j] == 255 and vis[i,j]==0):
                q = deque()
                q.append((i,j))
                comp = []

                # Pop until queue becomes empty
                while q:
                    x,y = q.popleft()

                    if (x>=0 and x<height and y>=0 and y<width and image[x,y] == 255 and vis[x,y]==0):
                        vis[x,y] = 1
                        comp.append((x,y))

                        # pushing down its eight neighbours
                        q.append((x+1,y))
                        q.append((x-1,y))
                        q.append((x,y+1))
                        q.append((x,y-1))
                        q.append((x+1,y+1))
                        q.append((x+1,y-1))
                        q.append((x-1,y+1))
                        q.append((x-1,y-1))

                if len(comp) > len(longest_component):
                    longest_component = comp

    img_res = np.zeros((height,width),np.uint8)

    for point in longest_component:
       img_res[point[0],point[1]] = 255

    return img_res

# User defined Bitwise Or opertion
def Bitwise_Or(image1,image2):

    h,w = image1.shape
    img_out = np.zeros((h,w),np.uint8)

    for i in range(h):
        for j in range(w):
            if (image1[i,j]==255 or image2[i,j]==255):
                img_out[i,j] = 255

    return img_out

# Hole Filling through Morphological Operations
def Fill_holes(image):

    # Creating a kernel of size 7 X 7
    kernel = np.ones((7,7),np.uint8)*255

    # Closing to fill small holes
    closed_image  = Closing(image,kernel)

    # Opening on closed_image to remove small objects
    opened_image = Opening(closed_image,kernel)

    # Performing dilation on opened_image to fill big holes
    dilated_image = dilation(opened_image,kernel)

    # Performing OR operation between original image and dilated image 
    img_out = Bitwise_Or(image,dilated_image)

    return img_out

# User defined , Different algorithm for hole filling
def hole_filling(img):
    height, width = img.shape
    new_img = np.zeros((height, width), np.uint8)
    for i in range(height):
        hit = 0
        hitb = 0
        j = 0
        x = width - 1
        while hit!=1 and j!=width:
            if img[i][j]==255:
                hit = 1
            j = j + 1

        while hitb!=1 and x!=0:
            if img[i][x]==255:
                hitb = 1
            x = x - 1

        for q in range(j-1,x+1):
            new_img[i][q] = 255

    return new_img

# Operation to get the final image from mask
def extract(Color_img,mask):
    height , width, channel = Color_img.shape
    new_img = np.zeros((height, width,channel), np.uint8)
    for i in range(height):
        for j in range(width):
            for c in range(channel):
                if mask[i][j]==255:
                    new_img[i][j][c] = Color_img[i][j][c]
                else:
                    new_img[i][j][c] = 255
    return new_img
   
# *********************************************************************** 

Original_image = cv2.imread('Assignment2.jpg')


HSV_image = cv2.cvtColor(Original_image,cv2.COLOR_BGR2HSV)


Binary_Image_mask = BinaryImage_inRange(HSV_image)


# Performing Dilation
kernel = np.ones((5,5),np.uint8)*255
dilated_img = dilation(Binary_Image_mask,kernel) # user defined


# Now finding the longest connected component
img_longestComponent = longest_connected_component(dilated_img)


# For filling Holes 

'''
Hole filling By Morphological Opeartions
img_out = Fill_holes(img_longestComponent)
img_holeFilled = Fill_holes(img_out)
'''

img_holeFilled = hole_filling(img_longestComponent)
img_holeFilled = dilation(img_holeFilled,kernel)




# Final resultant image
# For Final image take the And of Mask with Original Image
Final_image = extract(Original_image,img_holeFilled)
cv2.imshow('Final_Image',Final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


