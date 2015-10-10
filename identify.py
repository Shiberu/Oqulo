import cv2
import os
import serial
import sys
import csv
import matplotlib.pyplot as plt
import multiprocessing as mp

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
def calibrate(window=30.0, datapoints = 200):
    ser = serial.Serial('/dev/cu.usbmodem1411', 9600)
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

    ser.flushInput()
    print("Horizontal Calibration completed. Flushing.")

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
    ser = serial.Serial('/dev/cu.usbmodem1411', 9600)
    height, width, z = (486, 648, 3)

    # Calibration data
    (vert_avg, vert_min, vert_max, horiz_avg, horiz_min, horiz_max) = tup

    # Helper funciton to retrieve serial data
    def read_current_sight ():
        try:
            print("Reading current direction of sight:")
            index = 0
            arrx = []
            arry = []
            ser.flushInput()
            while index < window:
                point = ser.readline().split(',')
                arrx.append(float(point[0]))
                arry.append(float(point[1]))
                index+=1
            print("Current direction of sight recognized.")
            recX = sum(arrx)/float(window)
            recY = sum(arry)/float(window)
            return recX, recY
        except ValueError:
            print("Error while detecting current direction of sight:")
            return 5, 5 

    # Helper funciton to capture image with camera
    def get_current_vision ():
        print("Taking current vision:")
        ret_val, img = cam.read()
        img = cv2.flip(img, -1)
        img = cv2.resize(img, (0,0), fx=0.25, fy=0.25)
        print("Current vision taken.")
        return img

    while True:
        raw_input("Press Enter for recognition:")

        bmx, bmy = read_current_sight()
        if (bmx == 5 and bmy == 5):
            print("Hault; Error")
            continue

        img = get_current_vision()
        print(bmx)
        print(bmy)
        dx = horiz_max - bmx
        dy = vert_max - bmy
        print(horiz_max)
        print(vert_max)

        pixX = int(dx / float(horiz_max-horiz_min) * width)
        pixY = height - int(dy / float(vert_max-vert_min) * height)

        print(pixX)
        print(pixY)
        print(img.shape)

        for i in range(3):
            for x in range(-5, 6):
                for y in range(-5, 6):
                    if i==0:
                        img[pixY+y,pixX+x,i] = 255
                    else:
                        img[pixY+y,pixX+x,i] = 0


        cv2.imshow('Current Vision', img)

def identify():
    tup = calibrate()
    wait_for_request(tup)

if __name__ == '__main__':
    identify()