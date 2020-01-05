import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

import cv2


# Read in and grayscale the image

cap = cv2.VideoCapture('data/test_videos/solidYellowLeft.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

    # Define our parameters for Canny and apply
    low_threshold = 200
    high_threshold = 255
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    # Next we'll create a masked edges image using cv2.fillPoly()
    mask = np.zeros_like(edges)   
    ignore_mask_color = 255   

    # This time we are defining a four sided polygon to mask
    imshape = frame.shape

    X = [imshape[1]/8, 1.8*imshape[1]/5, 3.2*imshape[1]/5, 7*imshape[1]/8]
    X = [int(i) for i in X]

    Y = [imshape[0]-60, imshape[0]/1.5, imshape[0]/1.5, imshape[0]-60]
    Y = [int(i) for i in Y]

    vertices = np.array([[(X[0],Y[0]),(X[1], Y[1]), (X[2], Y[2]), (X[3],Y[3])]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    masked_edges = cv2.bitwise_and(edges, mask)

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = 1/2*(np.pi/180) # angular resolution in radians of the Hough grid
    threshold = 60     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 10  #minimum number of pixels making up a line
    max_line_gap = 15    # maximum gap in pixels between connectable line segments
    line_image = np.copy(frame)*0 # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = 0
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)

    # Iterate over the output "lines" and draw lines on a blank image
    print(lines)
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)

    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((edges, edges, edges)) 

    # Draw the lines on the edge image
    lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)
    lines_og = cv2.addWeighted(frame, 0.8, line_image, 1, 0)

    # Debugging
    # cv2.circle(lines_og,(X[1], Y[1]), 5, (255,255,255), -1)
    # cv2.circle(lines_og,(X[2], Y[2]), 5, (255,255,255), -1)
    # cv2.line(lines_og, (X[0], Y[0]),(X[1], Y[1]),(255,0,0),5)
    # cv2.line(lines_og, (X[2], Y[2]), (X[3], Y[3]),(255,0,0),5)


    cv2.imshow('frame',lines_og)
    cv2.imshow('lineedge',edges)
    cv2.imshow('maskededge',masked_edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

plt.imshow(lines_edges)
# plt.imshow(masked_edges)
plt.show()