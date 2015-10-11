import cv2
import os
import serial

def read_value(window=30.0, pix=100, readjustInterval=500):
	ser = serial.Serial('/dev/cu.usbmodem1421', 9600)
	index = 0
	horiz_list = [0]*int(window)
	vert_list = [0]*int(window)
	horiz_avg = 0
	vert_avg = 0


	# Initial polling
	while index < window:
		point = ser.readline().split(',')
		if len(point) <2:
			continue
		horiz_avg = (horiz_avg*window - horiz_list[0] + float(point[0]))/window
		vert_avg = (vert_avg*window - vert_list[0] + float(point[1]))/window
		
		horiz_list = horiz_list[1:]
		vert_list = vert_list[1:]

		horiz_list.append(float(point[0]))
		vert_list.append(float(point[1]))

		index+=1
	index = 0

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

		index+=1
		#if (index % pix == 0):
		#	img = take_picture()
		print '%.2f, %.2f' % (horiz_avg, vert_avg)
		
def take_picture():
	cam = cv2.VideoCapture(0)
	ret_val, img = cam.read()
	img = cv2.flip(img, -1)
	img = cv2.resize(img, (0,0), fx=0.25, fy=0.25)
	return img


#Capture function.
def show_webcam(mirror=False):
	cam = cv2.VideoCapture(0)
	index = 0
	while True:
		ret_val, img = cam.read()
		if mirror: 
			img = cv2.flip(img, -1)
		#img =cv2.fastNlMeansDenoisingColored(img,None,5,5,7,21)
		img = cv2.resize(img, (0,0), fx=0.25, fy=0.25)
		print img.shape
		cv2.imshow('Current Vision', img)
		ch = cv2.waitKey(1)
		if ch == 27: 
			break  # esc to quit
		elif ch == ord(' '):
			filename = 'Sample_' + str(index) + '.png'
			while (os.path.isfile(filename)):
				index += 1
				filename = 'Sample_' + str(index) + '.png'
			cv2.imwrite(filename, img)
			index += 1
	cv2.destroyAllWindows()

def main():
	read_value()
	#show_webcam(mirror=True)

if __name__ == '__main__':
	main()