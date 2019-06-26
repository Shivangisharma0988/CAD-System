
import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread("C:\\Users\\MOHIT VASHIST\\Desktop\\3.jpg")

blur = cv2.blur(img,(5,5))
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])

plt.show()
gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
kernel = np.ones((4,4),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=4)
sure_bg = cv2.bitwise_not(sure_bg)

plt.subplot(121),plt.imshow(sure_bg),plt.title('MORPHOLOGICAL OPERATION')
plt.xticks([]), plt.yticks([])

#lum_img = img[:,:,1]


#imgplot=plt.imshow(closing)
binary_map = (sure_bg> 0).astype(np.uint8)
connectivity = 4 # or whatever you prefer

output = cv2.connectedComponentsWithStats(binary_map, connectivity, cv2.CV_32S)
# Get the results
# The first cell is the number of labels
num_labels = output[0]
# The second cell is the label matrix
labels = output[1]
# The third cell is the stat matrix
stats = output[2]
# The fourth cell is the centroid matrix
centroids = output[3]

#connectedComponentswithStats yields every seperated component with information on each of them, such as size
#the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
sizes = stats[1:, -1]
num_labels = num_labels - 1

# minimum size of particles we want to keep (number of pixels)
#here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
min_size = 44892
#your answer image
img2 = np.zeros((labels.shape))
#imgplot=plt.imshow(img2)
#for every component in the image, you keep it only if it's above min_size
for i in range(0, num_labels):
    if sizes[i] >= min_size:
        img2[labels == i + 1] = 255

sub = sure_bg - img2

plt.subplot(122),plt.imshow(sub),plt.title('Reason Of Intrest(ROI)')
plt.xticks([]), plt.yticks([])
