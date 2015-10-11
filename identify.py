import cv2
import os
import serial
import sys
import csv
import matplotlib.pyplot as plt
import multiprocessing as mp
import scipy.spatial
import numpy as np

"""
    identify.py

    Final script that will be running on the laptop for serial data reading, camera
    image capture, and computer vision.

    Things to do:
    (1) Object boundary detection / object identification. Stater code is in grabCut.pyplot
    (2) After retrieving the object identification, send possible HTTP requests for Android
        application to retrieve.
"""

# Calibration function.
# Follow terminal outputs for directions.
# TODO: Remove duplicate code and improve formatting
def calibrate(window=30.0, datapoints = 200):
    ser = serial.Serial('/dev/cu.usbmodem1421', 9600)
    index = 0
    horiz_list = [0]*int(window)
    vert_list = [0]*int(window)
    horiz_avg = 0
    vert_avg = 0

    vert_max = -1
    vert_min = 9999
    horiz_max = -1
    horiz_min = 9999


    raw_input("Press Enter to start horizontal calibration...")

    # Initial polling for establishing moving average starting point.
    while index < window:
        point = ser.readline().split(',')
        if len(point) < 2:
            continue
        horiz_avg = (horiz_avg*window - horiz_list[0] + float(point[0]))/window
        vert_avg = (vert_avg*window - vert_list[0] + float(point[1]))/window
        
        horiz_list = horiz_list[1:]
        vert_list = vert_list[1:]

        horiz_list.append(float(point[0]))
        vert_list.append(float(point[1]))

        index+=1
    index = 0
    count = 0

    # Sample left to right (vice versa)
    while True:
        point = ser.readline().split(',')
        if len(point) <2:
            continue
        horiz_avg = (horiz_avg*window - horiz_list[0] + float(point[0]))/window
        vert_avg = (vert_avg*window - vert_list[0] + float(point[1]))/window
        
        horiz_list = horiz_list[1:]
        vert_list = vert_list[1:]

        horiz_list.append(float(point[0]))
        vert_list.append(float(point[1]))
        
        print '%.2f, %.2f' % (horiz_avg, vert_avg)
        if (horiz_avg < horiz_min):
            horiz_min = horiz_avg
        if (horiz_avg > horiz_max):
            horiz_max = horiz_avg

        if (datapoints == count):
            break
        count += 1

    print("Horizontal Calibration completed. Flushing.")
    ser.flushInput()
    if not ser.isOpen():
        ser.open()

    index = 0
    count = 0
    horiz_list = [0]*int(window)
    vert_list = [0]*int(window)
    horiz_avg = 0
    vert_avg = 0

    raw_input("Flushing completed. Press Enter to start vertical calibration...")

    # Initial polling for establishing moving average starting point.
    while index < window:
        point = ser.readline().split(',')
        if len(point) < 2:
            continue
        horiz_avg = (horiz_avg*window - horiz_list[0] + float(point[0]))/window
        vert_avg = (vert_avg*window - vert_list[0] + float(point[1]))/window
        
        horiz_list = horiz_list[1:]
        vert_list = vert_list[1:]

        horiz_list.append(float(point[0]))
        vert_list.append(float(point[1]))

        index+=1
    index = 0

    # Sample top to bottom (vice versa)
    while True:
        point = ser.readline().split(',')
        if len(point) <2:
            continue
        horiz_avg = (horiz_avg*window - horiz_list[0] + float(point[0]))/window
        vert_avg = (vert_avg*window - vert_list[0] + float(point[1]))/window
        
        horiz_list = horiz_list[1:]
        vert_list = vert_list[1:]

        horiz_list.append(float(point[0]))
        vert_list.append(float(point[1]))
        
        print '%.2f, %.2f' % (horiz_avg, vert_avg)
        if (vert_avg < vert_min):
            vert_min = vert_avg
        if (vert_avg > vert_max):
            vert_max = vert_avg

        if (datapoints == count):
            break
        count += 1

    vert_avg = (vert_min+vert_max)/2.0
    horiz_avg = (horiz_min+horiz_max)/2.0

    print "Calibration completed."
    print "results:"
    print ("Vert min: " + str(vert_min))
    print ("Vert avg: " + str(vert_avg))
    print ("Vert max: " + str(vert_max))
    print ("Horiz min: " + str(horiz_min))
    print ("Horiz avg: " + str(horiz_avg))
    print ("Horiz max: " + str(horiz_max))

    return vert_avg, vert_min, vert_max, horiz_avg, horiz_min, horiz_max

# Polling function that will take picture and serial reading for object detection.
def wait_for_request(tup, window=30):
    # Initialize necessary data sources
    cam = cv2.VideoCapture(0)
    ser = serial.Serial('/dev/cu.usbmodem1421', 9600, timeout=5)

    # This is the camera image size we are expecting.
    height, width, z = (486, 648, 3)

    # Retrieve the trainImages
    targetPath = './targets'
    trainImages = [[os.path.splitext(f)[0] ,cv2.imread('targets/' + f,0)] for f in os.listdir(targetPath) if os.path.isfile(os.path.join(targetPath, f)) and f != ".DS_Store"]

    # Calibration data
    (vert_avg, vert_min, vert_max, horiz_avg, horiz_min, horiz_max) = tup

    # Initiate SIFT detector
    sift = cv2.SIFT()

    for ind in range(len(trainImages)):          # Object Name and trainImage
        [objectName, train] = trainImages[ind]
        if (objectName == '.DS_Store'):
            continue
        
        print("Importing image for " + objectName + "...")
        
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(train,None)

        trainImages[ind].append(kp1)
        trainImages[ind].append(des1)
        
    print("Import completed.")


    # Helper funciton to retrieve serial data
    def read_current_sight ():
        try:
            print("Reading current direction of sight:")
            index = 0
            arrx = []
            arry = []
            ser.flushInput()
            if not ser.isOpen():
                ser.open()
            while index < window:
                point = ser.readline().split(',')
                arrx.append(float(point[0]))
                arry.append(float(point[1]))
                index+=1
            print("Current direction of sight recognized.")
            recX = sum(arrx)/float(window)
            recY = sum(arry)/float(window)
            ser.close()
            return recX, recY
        except: # Catch all exceptions; stop from crashing the code and prevent recalibration
            print("Error while detecting current direction of sight:")
            return 5, 5 

    # Helper funciton to capture image with camera
    def get_current_vision ():
        print("Taking a photo from camera:")
        ret_val, img = cam.read()
        img = cv2.flip(img, -1)
        img = cv2.resize(img, (0,0), fx=0.25, fy=0.25)

        MIN_MATCH_COUNT = 10

        retObjs = []

        print("Recognizing features...")
        # find the keypoints and descriptors with SIFT
        kp2, des2 = sift.detectAndCompute(img,None)

        print("Identifying...")
        for [objectName, train, kp1, des1] in trainImages:          # Object Name and trainImage
            print("Looking for " + objectName + "...")

            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks = 50)

            flann = cv2.FlannBasedMatcher(index_params, search_params)

            matches = flann.knnMatch(des1,des2,k=2)

            # store all the good matches as per Lowe's ratio test.
            good = []
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    good.append(m)

            if len(good)>MIN_MATCH_COUNT:
                print "Found!"
                src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                matchesMask = mask.ravel().tolist()

                h,w = train.shape
                pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                dst = cv2.perspectiveTransform(pts,M)

                retObjs.append([objectName, dst])
            else:
                print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
                matchesMask = None

        # You want to locate the object boundaries only after all object search has been completed.
        for [name, points] in retObjs:
            cv2.polylines(img,[np.int32(points)],True,255,3, 16)

        print("Current vision taken, marked, and identified")
        return img, retObjs

    index = 0
    count = 0
    horiz_list = [0]*int(window)
    vert_list = [0]*int(window)
    horiz_avg = 0
    vert_avg = 0

    if (not ser.isOpen()):
        ser.open()

    # Initial polling for establishing moving average starting point.
    while index < window:
        point = ser.readline().split(',')
        if len(point) < 2:
            continue
        horiz_avg = (horiz_avg*window - horiz_list[0] + float(point[0]))/window
        vert_avg = (vert_avg*window - vert_list[0] + float(point[1]))/window
        
        horiz_list = horiz_list[1:]
        vert_list = vert_list[1:]

        horiz_list.append(float(point[0]))
        vert_list.append(float(point[1]))

        index+=1
    
    FRAME_UPDATE = 50
    count = 0

    while True:
        point = ser.readline().split(',')
        if len(point) <2:
            continue
        horiz_avg = (horiz_avg*window - horiz_list[0] + float(point[0]))/window
        vert_avg = (vert_avg*window - vert_list[0] + float(point[1]))/window
        
        horiz_list = horiz_list[1:]
        vert_list = vert_list[1:]

        horiz_list.append(float(point[0]))
        vert_list.append(float(point[1]))
        
        
        
        dx = horiz_max - horiz_avg
        dy = vert_max - vert_avg

        pixX = int(dx / float(horiz_max-horiz_min) * width)
        pixY = height - int(dy / float(vert_max-vert_min) * height)

        if (count == FRAME_UPDATE):
            # Perform image processing fist; this will take longer
            img, retObjs = get_current_vision()
            for i in range(3):
                for x in range(-5, 6):
                    for y in range(-5, 6):
                        if i==1:
                            img[pixY+y,pixX+x,i] = 255
                        else:
                            img[pixY+y,pixX+x,i] = 0

            cv2.imshow('Current Vision', img)
            cv2.waitKey(1)

            # Approximate the closest object by looking for closest point.
            spt = np.array([pixX, pixY])
            min_dst = 9999
            predObj = None
            for [name, points] in retObjs:
                for p in points:
                    curr_dst = scipy.spatial.distance.euclidean(p, spt)
                    if (curr_dst < min_dst):
                        min_dst = curr_dst
                        predObj = name
        
            if predObj is not None:
                print("\n\n\n\n************************************************\n"+
                    "Are you looking at: " + predObj + "\n\n\n\n************************************************\n\n")
            else:
                print("\n\n\n\n************************************************\n"+
                    "Nothing discovered. Retry! \n\n\n\n************************************************\n\n")
            count = 0
        else:
            count += 1
            continue
    """
    while True:
        raw_input("Press Enter for recognition:")

        # Perform image processing fist; this will take longer
        img, retObjs = get_current_vision()

        # Read in data from Arduino Serial
        bmx, bmy = read_current_sight()
        print("Current direction of sight: " + str(bmx) + "," + str(bmy))
        if (bmx == 5 and bmy == 5):
            print("Error: Skipping this instance; possible sensor data misread")
            continue
        
        dx = horiz_max - bmx
        dy = vert_max - bmy

        pixX = int(dx / float(horiz_max-horiz_min) * width)
        pixY = height - int(dy / float(vert_max-vert_min) * height)

        print("Mapped sight location: " + str(pixX) + "," + str(pixY))

        try:
            for i in range(3):
                for x in range(-5, 6):
                    for y in range(-5, 6):
                        if i==1:
                            img[pixY+y,pixX+x,i] = 255
                        else:
                            img[pixY+y,pixX+x,i] = 0

            cv2.imshow('Current Vision', img)
            raw_input("Press Enter to proceed to identificaiton:")

        except: # Catch all exceptions; misread value can cause index error.
            print("Error: Skipping this instance; possible sensor data misread")

        # Approximate the closest object by looking for closest point.
        spt = np.array([pixX, pixY])
        min_dst = 9999
        predObj = None
        for [name, points] in retObjs:
            for p in points:
                curr_dst = scipy.spatial.distance.euclidean(p, spt)
                if (curr_dst < min_dst):
                    min_dst = curr_dst
                    predObj = name
        
        if predObj is not None:
            print("\n\n\n\n************************************************\n"+
                "Are you looking at: " + predObj + "\n\n\n\n************************************************\n\n")
        else:
            print("\n\n\n\n************************************************\n"+
                "Nothing discovered. Retry! \n\n\n\n************************************************\n\n")
    """

def identify():
    tup = calibrate()
    wait_for_request(tup)

if __name__ == '__main__':
    identify()