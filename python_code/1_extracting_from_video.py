''' How to use these scripts: Please run these scripts in spyder, as they were made in spyder and I can't guarantee they will work from the command line, jupyter notebook etc. See the git repository readme for instructions on how to install the computing environment from the .yml file
    
    This script finds the region of interest i.e. where there are numbers changing in the video
    It then takes the numbers and exports a black & white image of them to a folder
    There is no machine learning happening at this point. After running this script, someone must sort the black and white images into the appropriate folder
    This task has been done for you for about 300-400 images of numbers since it takes about 10 minutes to sort
    To skip sorting the images, please DELETE outputs\extracted_training_data and unzip "exampleML\outputs.zip". The sorting should be done for you
    
    If you wish to do the sorting, please navigate to 'exampleML\outputs\extracted_training_data' after running this script
    Drag each image into the folder named with the digit shown. i.e. if you see a png which is a black & white 3, drag this into the folder outputs\extracted_training_data\3
    Be careful when sorting the 9s and gs
    Once there are no images remaining in the parent folder, each subfolder should have some numbers in it and you can move on to the second script "2_model_training.py"
'''

##############################################################################################################################
############################################################# SETTINGS #######################################################
##############################################################################################################################

# This section defines thresholds and parameters for interpreting the video. These things exist so that you can tweak the approach to other videos, so it is safest to leave it alone at first. 


wait_time = 10 # for cropping threshold
text_diff_threshold=2.5
default_lightness_min=170 # 125 for bad quality # for identifying white text through lightness
default_lightness_max =255
default_hue_min = 0 #50 for bad quality 
default_hue_max = 179
default_saturation_min = 0
default_saturation_max = 255 # 19 for bad quality 
show_video=True
limit_FOV=False # True for bad quality 
(FOVx,FOVy,FOVw,FOVh)=(915,10,700,250) # bounding rect if find_ROI_automatically
find_ROI_automatically=True # Keep as true. not implemented
import os; working_in= os.path.dirname(os.path.realpath(__file__))
package_directory=working_in[0:-12]
save_images=True
print_result=True
rotate_180=False # True for bad quality 
text_thresholding_percentile = 99 # 65  for bad quality 
black_on_white_or_white_on_black = 'white_on_black'#'black_on_white' #specify text colour vs backround
transform_projection_angle=90 # 90 = birds eye /  no transformation. Stretches or squashes image at the top or bottom. This can be a positive or negative number
transform_skew_angle=0# 0 = no skew. Skew angle in the x direction. Skews the image diagonally.
slant_proportion=6 # The 'transform' is a bit of trig and a bit of empirical approximation. thinking about it, the x and y vertices of a spinning square form two sine waves if you track the x and y position whilst it spins to be completely flat and then flipped over. The sine wave in x will covary with the sine wave in y, but it will be much smaller in amplitude depending on the shape. The exact relationship between the x and y amplitudes is hard to put my finger on but I have put this number in so it can be varied (basically an empirical relationship)
anticipated_aspect_ratio = 3 # describes bounding box aspect ratio we are looking for. potentially font sensitive. I am doing this because when identifying ROI over a few seconds it will likely pick up on a patch of pixels that crops out anything over 999g. based on the aspect ratio of 2.209 for a video of three digits and a g, i.e. 239g, 673g, ----. multiply by 5 and divide by 4 to get anticipated aspect ratio of anything in the kilo range
zoomcut=0 # how much zoom out on either side.  0 = full size
zoomout= 1-zoomcut # From above, 1 = full size, 0.5 = infinitesimally small (zero) 
#vidfilein=0 # means use webcam as input video.
vidfilein=package_directory+r'\inputs\training_video\phone_video_of_display_1_minute.mp4'
                            
                            
########################################################################################################################################################################################################################
############################################################# IMPORTS AND FUNCTIONS #######################################################
########################################################################################################################################################################################################################

#Imports and UD functions
import cv2, os, datetime, time
import numpy as np
import imutils
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import pandas as pd
from skimage import img_as_ubyte
from skimage.color import rgb2gray
from keras.models import load_model

# Finds the region of interest by looking for consistent motion in the frame over a set number of frames. It picks an area containing the top 95% of variability in the frame
# This function is delicate, and needs to update the moving average very quickly for best results, effectively comparing each frame with the frames a fraction of a second before
def find_roi(limit_FOV,vidfilein,capt,frame1, anticipated_aspect_ratiof, capture_frames = 400, start_time = time.time()):
       
    # Start the rolling average using first frame
    avg1 = np.float32(frame1)
    diffavg1 = np.float32(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY))
    diffavg1[:]=0 
    
    # Watch the video and identify moving objects in the video
    i=0
    # Load a few seconds of footage and identify where the moving numbers are. It would be good to also identify here whether it is upside down or not somehow
    while( i < capture_frames):  
        i=i+1
        if i % 10 == 1: #Only update the running average every 10 frames 
           cv2.accumulateWeighted(frame1,avg1,0.9) # Update avg1. 0.04 equals 25 frame smoothing so that each second the background is 'tracked' not a fixed estimate
           medianFrame = np.uint8(avg1)
           diff = cv2.absdiff(frame1, medianFrame)
           gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
           cv2.accumulateWeighted(gray,diffavg1,0.01) # Update avg1. 0.04 equals 25 frame smoothing so that each second the background is 'tracked' not a fixed estimate

        # display the Region of Interest / ROI being detected
        cv2.imshow("Security Feed", diffavg1) # can be frame, frameDelta, gray, thresh
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
            break

        _, frame1 = capt.read() # Move on to next frame
        if frame1 is None:
            break
        if limit_FOV:
            frame1=frame1[FOVy:FOVy+FOVh, FOVx:FOVx+FOVw]
            
    # threshold is variable. We are grabbing the top 1% of mean frame difference pixels (which should be where the numbers appear). Threshold, Invert, Dilate away thin lines from the vibrations in the frame, uninvert
    _, thresh = cv2.threshold(diffavg1, np.percentile(diffavg1,text_thresholding_percentile), 255, cv2.THRESH_BINARY)    
    if black_on_white_or_white_on_black == 'white_on_black':
        dilated_inverted= 255 - cv2.dilate(255 - thresh, None, iterations=1).astype(np.uint8)
    if black_on_white_or_white_on_black == 'black_on_white':
        dilated_inverted= cv2.dilate(255 - thresh, None, iterations=1).astype(np.uint8)

    #redilate to merge into one rounded large blob of where the moving numbers are with a substantial 20 dilations
    dilated_inverted=cv2.dilate(dilated_inverted, None, iterations=20)

    # Contour the potential regions of interest
    cnts = cv2.findContours(dilated_inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    # predefine an output in case of null result
    (fxi, fyi, fwi, fhi) = (10,10,10,10)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # loop over the contours. This will find one region of activity that is likely to be the numbers but if there are two regions it will just use the first one it finds. We will want to update this if monitoring two belts.
    for c in cnts:
        if cv2.contourArea(c) < (frame_width * frame_height)/200: # Too small - unlikely to be changing numbers
            continue
        if cv2.contourArea(c) > (frame_width * frame_height)*0.5: # These numbers will be dependant on the frame dimensions, area of 1 million is half of a 1080*1920 image
            continue
        (fxi, fyi, fwi, fhi) = cv2.boundingRect(c) # region of interest

        # One issue was that the g (gram) symbol at the end of the numbers doesn't change, and so doesn't get 'seen' as a varying number. To get around this I expanded the region of interest to be a little bit wider than it otherwise would be
        fhi = int(fhi+(fhi*0.2)) # expand ROI to accomodate the g symbol if it isn't in the region of movement... height
        fyi = int(fyi+(fhi*0.2)/2) # expand ROI to accomodate the g symbol if it isn't in the region of movement... top left coord
        continue

    # Define bounds and check we have no negative indices if operating at edge of the frame
    additional_width_from_h=((anticipated_aspect_ratiof*fhi)-fwi)/2
    minxi = int(fxi-additional_width_from_h)
    maxxi = int(fxi+fwi+(additional_width_from_h))
    maxyi=fyi+fhi
    minyi=fyi
    if minxi<0:
        minxi=0
    if maxxi>frame_width:
        maxxi=frame_width
    if maxyi>frame_height:
        maxyi=frame_height
    if minyi<0:
        minyi=0

    cv2.destroyAllWindows()
    return(fxi, fyi, fwi, fhi, additional_width_from_h, minxi, maxxi, minyi, maxyi)



def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect


def resize2SquareKeepingAspectRation(img, size, interpolation):
  h, w = img.shape[:2]
  c = None if len(img.shape) < 3 else img.shape[2]
  if h == w: return cv2.resize(img, (size, size), interpolation)
  if h > w: dif = h
  else:     dif = w
  x_pos = int((dif - w)/2.)
  y_pos = int((dif - h)/2.)
  if c is None:
    mask = np.zeros((dif, dif), dtype=img.dtype)
    mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
  else:
    mask = np.zeros((dif, dif, c), dtype=img.dtype)
    mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]
  return cv2.resize(mask, (size, size), interpolation)


def four_point_transform(image, pts, zoomfactor):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
    	[maxWidth*(1-zoomfactor), maxHeight*(1-zoomfactor)],
    	[maxWidth*zoomfactor, maxHeight*(1-zoomfactor)],
    	[maxWidth*zoomfactor,maxHeight*zoomfactor],
    	[maxWidth*(1-zoomfactor), maxHeight*zoomfactor]], dtype = "float32")
       	# compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


def nukedir(dir):
    if dir[-1] == os.sep: dir = dir[:-1]
    files = os.listdir(dir)
    for file in files:
        if file == '.' or file == '..': continue
        path = dir + os.sep + file
        if os.path.isdir(path):
            nukedir(path)
        else:
            os.unlink(path)
    os.rmdir(dir)


# function copy-pasted from https://stackoverflow.com/a/14178717/744230
def find_coeffs(source_coords, target_coords):
    matrix = []
    for s, t in zip(source_coords, target_coords):
        matrix.append([t[0], t[1], 1, 0, 0, 0, -s[0]*t[0], -s[0]*t[1]])
        matrix.append([0, 0, 0, t[0], t[1], 1, -s[1]*t[0], -s[1]*t[1]])
    A = np.matrix(matrix, dtype=np.float)
    B = np.array(source_coords).reshape(8)
    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)



########################################################################################################################################################################################################################
########################################################## PROCESSING SETUP #######################################################
########################################################################################################################################################################################################################

# Trig approximation for tansformation. I tried to use cameratransform package but it wasn't working and documentation explains it poorly
# thinking about it, the x and y vertices of a spinning square form two sine waves if you track the x and y position whilst it spins to be completely flat and then flipped over. The sine wave in x will covary with the sine wave in y, but it will be much smaller in amplitude depending on the aspect ratio. The exact relationship between the x and y amplitudes is hard to put my finger on. And there is also offset to consider
x_trig_skew=np.sin(np.deg2rad(transform_skew_angle))
trig_ratio_of_nadir_to_focal_point=1/np.sin(np.deg2rad(transform_projection_angle))
x_trig_stretch=1+(trig_ratio_of_nadir_to_focal_point-1)/slant_proportion


# Setup and identification of parameters in the input video
outputs_directory=package_directory+r'\outputs\extracted_training_data\\'

# Tries 2 times to delete and remake the temp directory
try:
    nukedir(outputs_directory) # should check if exists first
    time.sleep(4)
    os.mkdir(outputs_directory) # should check if exists first
except:
    try:
        nukedir(outputs_directory) # should check if exists first
        time.sleep(4)
        os.mkdir(outputs_directory) # should check if exists first
    except:
        0
time.sleep(1)

os.mkdir(outputs_directory)

if save_images:
    for dr in ["0","1","2","3","4","5","6","7","8","9","g"]:
        os.mkdir(outputs_directory+dr+"\\")
os.chdir(outputs_directory)


# Start capture. 
cap = cv2.VideoCapture(vidfilein) # Open mp4 file
_, frame1 = cap.read()
if limit_FOV:
    frame1=frame1[FOVy:FOVy+FOVh, FOVx:FOVx+FOVw]
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
numframes=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


# Find ROI then crop
avg1 = np.float32(frame1)
if find_ROI_automatically:
    (xi, yi, wi, hi, additional_width_from_h, minxi, maxxi, minyi, maxyi) = find_roi(limit_FOV,vidfilein,cap,frame1,anticipated_aspect_ratio)

avg1=avg1[minyi:maxyi, minxi:maxxi, 0]
avg1 = avg1.astype(np.float32)

# define pts. this is used to transform the image if some kind of projection is used.
top_add_to_either_side=((wi*x_trig_stretch)-wi)/2
width_from_aspect=wi+(additional_width_from_h*2)
pts=np.array([(0+((hi/2)*x_trig_skew), 0), (width_from_aspect, 0), (width_from_aspect-top_add_to_either_side-((hi/2)*x_trig_skew),hi), (0+top_add_to_either_side, hi)], dtype="float32")

# Set beginning of loop parameters to 0 / empty
i=0; savecounter=0; diffcurrent=0; diffprevious=0; diffbeforeprevious=0
meandifframe=2; meandeviation_from_prev=2 ; digitlist = []; timeslist = []; viewlist = []
kernel5 = np.ones((5,5),np.float32)/25

# Empty window. Some kind of window is needed to respond to pressed key
cv2.namedWindow('null', cv2.WINDOW_NORMAL)

# This section gives an opportunity to change how the text is thresholded based on colour, which should make it more malleable
# Taken from stack overflow https://stackoverflow.com/questions/22588146/tracking-white-color-using-python-opencv
def nothing(x):
    pass
# Create a window
cv2.namedWindow('This window can be used to test thresholds. Please ignore it and wait for it to close')
# create trackbars for color change
cv2.createTrackbar('HMin','This window can be used to test thresholds. Please ignore it and wait for it to close',0,179,nothing) # Hue is from 0-179 for Opencv
cv2.createTrackbar('SMin','This window can be used to test thresholds. Please ignore it and wait for it to close',0,255,nothing)
cv2.createTrackbar('LMin','This window can be used to test thresholds. Please ignore it and wait for it to close',0,255,nothing)
cv2.createTrackbar('HMax','This window can be used to test thresholds. Please ignore it and wait for it to close',0,179,nothing)
cv2.createTrackbar('SMax','This window can be used to test thresholds. Please ignore it and wait for it to close',0,255,nothing)
cv2.createTrackbar('LMax','This window can be used to test thresholds. Please ignore it and wait for it to close',0,255,nothing)
# Set default value for MAX HSV trackbars.
cv2.setTrackbarPos('HMin', 'This window can be used to test thresholds. Please ignore it and wait for it to close', default_hue_min)
cv2.setTrackbarPos('SMin', 'This window can be used to test thresholds. Please ignore it and wait for it to close', default_saturation_min)
cv2.setTrackbarPos('HMax', 'This window can be used to test thresholds. Please ignore it and wait for it to close', default_hue_max)
cv2.setTrackbarPos('SMax', 'This window can be used to test thresholds. Please ignore it and wait for it to close', default_saturation_max)
cv2.setTrackbarPos('LMax', 'This window can be used to test thresholds. Please ignore it and wait for it to close', default_lightness_max)
cv2.setTrackbarPos('LMin', 'This window can be used to test thresholds. Please ignore it and wait for it to close', default_lightness_min)
# Initialize to check if HSV min/max value changes
hMin = sMin = lMin = hMax = sMax = lMax = 0
phMin = psMin = plMin = phMax = psMax = plMax = 0
lMin = default_lightness_min
lMax = default_lightness_max
hMin = default_hue_min
hMax = default_hue_max
sMin = default_saturation_min
sMax = default_saturation_max


output = frame1
start_time = time.time()
while(time.time()<(start_time+wait_time)):
    # get current positions of all trackbars
    hMin = cv2.getTrackbarPos('HMin','This window can be used to test thresholds. Please ignore it and wait for it to close')
    sMin = cv2.getTrackbarPos('LMin','This window can be used to test thresholds. Please ignore it and wait for it to close')
    lMin = cv2.getTrackbarPos('SMin','This window can be used to test thresholds. Please ignore it and wait for it to close')
    hMax = cv2.getTrackbarPos('HMax','This window can be used to test thresholds. Please ignore it and wait for it to close')
    sMax = cv2.getTrackbarPos('LMax','This window can be used to test thresholds. Please ignore it and wait for it to close')
    lMax = cv2.getTrackbarPos('SMax','This window can be used to test thresholds. Please ignore it and wait for it to close')
    # Set minimum and max HSV values to display
    lower = np.array([hMin, sMin, lMin])
    upper = np.array([hMax, sMax, lMax])
    # Create HSV Image and threshold into a range.
    hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HLS)
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(frame1,frame1, mask= mask)
    # Print if there is a change in HSV value
    if( (phMin != hMin) | (psMin != sMin) | (plMin != lMin) | (phMax != hMax) | (psMax != sMax) | (plMax != lMax) ):
#        print("(hMin = %d , sMin = %d, lMin = %d), (hMax = %d , sMax = %d, lMax = %d)" % (hMin , sMin , lMin, hMax, sMax , lMax))
        phMin = hMin
        psMin = sMin
        plMin = lMin
        phMax = hMax
        psMax = sMax
        plMax = lMax
    # Display output image
    cv2.imshow('This window can be used to test thresholds. Please ignore it and wait for it to close',output)
    # Wait longer to prevent freeze for videos.
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()


########################################################################################################################################################################################################################
########################################################## PROCESSING MAIN LOOP #######################################################
########################################################################################################################################################################################################################

# Begin interpreting
while cap.isOpened():
    if frame1 is None:
        break

    i=i+1 # only needed if playing back a video (not required for camera)
    key = cv2.waitKey(1) & 0xFF
    if key != 255:
        if find_ROI_automatically:
            print('updating roi')
            (xi, yi, wi, hi, additional_width_from_h, minxi, maxxi, minyi, maxyi) = find_roi(limit_FOV,vidfilein,cap,frame1,anticipated_aspect_ratio)
            avg1 = np.float32(frame1)
            avg1=avg1[minyi:maxyi, minxi:maxxi, 0]
            avg1 = avg1.astype(np.float32)
            top_add_to_either_side=((wi*x_trig_stretch)-wi)/2
            width_from_aspect=wi+(additional_width_from_h*2)
            pts=np.array([(0+((hi/2)*x_trig_skew), 0), (width_from_aspect, 0), (width_from_aspect-top_add_to_either_side-((hi/2)*x_trig_skew),hi), (0+top_add_to_either_side, hi)], dtype="float32")
            cv2.namedWindow('null', cv2.WINDOW_NORMAL) # window is needed to respond to pressed key. reinitialise
        continue

    else: #motion detect as normal
        savecounter=savecounter+1    
        # _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        frame1=frame1[minyi:maxyi, minxi:maxxi]
        hls = cv2.cvtColor(frame1, cv2.COLOR_BGR2HLS)
        mask = cv2.inRange(hls, lower, upper)
        thresh = cv2.erode(cv2.dilate(mask, None, iterations=3), None, iterations=3)
        threshy = thresh.astype(np.float32).copy() 
        cv2.accumulateWeighted(threshy,avg1,0.5) # Update avg1, less than 0.5 is not neccessary as 0.5 is a good quick update. 0.04 equals 25 frame smoothing so that each second the background is 'tracked' not a fixed estimate
        medianFrame = np.uint8(avg1)
        diff = cv2.absdiff(thresh, medianFrame)


#        _, thresh = cv2.threshold(gray, np.percentile(gray,text_thresholding_percentile*1), 255, cv2.THRESH_BINARY)
        # Here we ask: how much are pixels different from the previous frame?
        #diffbeforeprevious=diffprevious
        diffprevious=diffcurrent
        meandifframe = ((1000*meandifframe)+diffprevious)/1001 # big 5000 frame rolling window to identify mean difference
        meandeviation_from_prev = ((1000*meandeviation_from_prev)+np.mean(diff))/1001 # big 500 frame window to identify mean difference
        #viewlist.append(abs(diffprevious-diffbeforeprevious))
        diffcurrent=np.mean(diff)
        print(meandeviation_from_prev)
        if savecounter>5: # Don't interpret if we've interpreted a number in the last 5 frames (FPS dependent. at 30fps this is a wait of 0.16 seconds)
            if abs(diffprevious)>text_diff_threshold:#np.logical_or(abs(diffprevious)>2.5,  abs(diffbeforeprevious)>2.5) :#(meandeviation_from_prev): # Proceed if the difference from the frame just before is substantial enough
                if diffcurrent<diffprevious: # Take a snap once the frames have stabilised. For a rolling video there is a 'peak' in the difference about 2 frames after the number starts to change 
                    if transform_projection_angle!=90 or transform_skew_angle!=0:
                        thresh = four_point_transform(thresh, pts,zoomout)
                    if rotate_180:
                        thresh = np.rot90(np.rot90(thresh))


                    framedtime=datetime.datetime.today().strftime('%Y_%m_%d_%H_%M_%S_%f')
                    savecounter = 0 # reset counter
    
                    framey = thresh.astype(np.uint8)
                    # framey = 255-frameout
                    cnts = cv2.findContours(framey, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    cnts = imutils.grab_contours(cnts)
                    if len(cnts)>0:
                        cnts = sort_contours(cnts)
                        cnts = imutils.grab_contours(cnts)
                    i_n=0
                    digits = []

                    # Loop over the contours
                    for c in cnts:
                        # if the contour is too small, ignore it
                        if cv2.contourArea(c) > 1000:
                            i_n=i_n+1
                            (x, y, w, h) = cv2.boundingRect(c)
                            frame=framey[y:y+h, x:x+w]
                            frame = resize2SquareKeepingAspectRation(frame, 28, cv2.INTER_AREA)
                            im_gray = rgb2gray(frame)#convert original to gray image
                            img_gray_u8 = img_as_ubyte(im_gray)# convert grey image to uint8
                            _,im_bw = cv2.threshold(img_gray_u8, 28, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                            cv2.imwrite(outputs_directory+"motion_detected_at_"+str(framedtime)+str(i_n)+".png",im_bw)                            

                    
        _, frame1 = cap.read() # Move on to next frame
            
        if show_video:
            cv2.imshow("Security Feed", thresh) # can be frame, frameDelta, gray, thresh
        # display the ROI being detected
        if key == ord("q"):
            break
        if limit_FOV:
            frame1=frame1[FOVy:FOVy+FOVh, FOVx:FOVx+FOVw]


cv2.destroyAllWindows(); cap.release()#; out.release()

# For reproducibility, export a snapshot of the environment in which this script was run (to be completed)
# import yaml
# yaml.dump()  
