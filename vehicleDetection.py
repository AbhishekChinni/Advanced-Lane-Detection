import glob
import cv2
import numpy as np
from skimage.feature import hog
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import pickle

datasetPath = "../datasets/"
processed = 0
#xscaler = 0 # will be initialized later
#svc = 0 # will be initialized later while loading from pickle file
print("Loading model..")
svc = pickle.load(open('car_classifier_model.sav', 'rb'))
xscaler = pickle.load(open('xscaler.sav', 'rb'))


def getfeatures(img):
	global processed

	# img is 64*64*3
	#print(type(img))

	# get color histogram (hist for each channel)
	chist0 =  np.histogram(img[:,:,0], bins=32)[0]
	chist1 =  np.histogram(img[:,:,1], bins=32)[0]
	chist2 =  np.histogram(img[:,:,2], bins=32)[0]
	chist = np.concatenate((chist0,chist1,chist2))
	#print("chist0 shape:", chist0.shape) #32
	#print("chist shape:", chist.shape) #96

	# get hog features
	visualizefl = False
	hf0 = hog(img[:,:,0],visualise=visualizefl) # 1d array of 2916
	hf1 = hog(img[:,:,1])
	hf2 = hog(img[:,:,2])
	
	#print("hf0 shape: ",hf0.shape)
	#print("himg shape:", himg.shape)
	#plt.imshow(himg, interpolation='nearest')
	#plt.show()
	
	
	print("Processed=",processed)
	processed = processed + 1
	return np.concatenate((hf0,hf1,hf2,chist))

def getWindows(xstart,xend,ystart,yend,windowsize,overlap=(0.8,0.8)):

	pixelspershift = [int(windowsize[0] * overlap[0]) , int(windowsize[1] * overlap[1])]
	totlen = [xend - xstart, yend - ystart]
	windowct = [int((totlen[0]-windowsize[0])//pixelspershift[0])+1, int((totlen[1]-windowsize[1])//pixelspershift[1])+1]
	leftpixels = totlen[1] - windowsize[1] - (windowct[1]-1) * pixelspershift[1]
	#print(windowct)
	#print(xstart, xend, ystart, yend, windowsize, overlap)
	windows = []
	for xct in range(windowct[0]):
		for yct in range(windowct[1]):
			topleft = (xstart+xct*pixelspershift[0], leftpixels+ystart+yct*pixelspershift[1])
			bottomright = (topleft[0]+ windowsize[0],topleft[1]+windowsize[1])
			windows.append((topleft, bottomright))

	return windows

def classifyWindows(img,xstart,xend,ystart,yend,windowsize,overlap=(0.8,0.8)):

	windows = getWindows(xstart,xend,ystart,yend,windowsize,overlap)
	print(windows[0])
	# get image portions of the windows and resize to 64*64
	windowimgs = [cv2.resize(img[w[0][1]:w[1][1], w[0][0]:w[1][0]], dsize =(64,64)) for w in windows]
	winfeatures = [getfeatures(im) for im in windowimgs]
	winfeatures = np.asarray(winfeatures).astype(np.float64)
	print("windows len:=",len(windows))
	testx = xscaler.transform(winfeatures)
	prediction = svc.predict(testx)
	predictionprobability = svc.predict_proba(testx)

	carwindows = [windows[i] for i in range(len(windows)) if prediction[i]==1]
	carwinprobability = [predictionprobability[i][0] for i in range(len(windows)) if prediction[i]==1]
	print(prediction)
	print(len(carwindows))
	return carwindows, carwinprobability

def getcolor(prob):

	if prob >= 0.7:
		return (0,255,0)
	else: return (255,0,0)

def findcars(img):

	imgcopy = np.copy(img)
	threshold = 0.4
	# big window for cars which are near
	carwindows1, carwinprobability1 = classifyWindows(img,0,img.shape[1],380,img.shape[0],(128,128))
	found = False
	for i in range(len(carwindows1)):
		if carwinprobability1[i] > threshold:
			cv2.rectangle(imgcopy, carwindows1[i][0], carwindows1[i][1], getcolor(carwinprobability1[i]), 6)
		found = True
	# small window for distant cars
	carwindows1, carwinprobability1 = classifyWindows(img,0,img.shape[1],400,500,(64,64))

	for i in range(len(carwindows1)):
		if carwinprobability1[i] > threshold:
			cv2.rectangle(imgcopy, carwindows1[i][0], carwindows1[i][1], getcolor(carwinprobability1[i]), 6)
		found = True
	return imgcopy,found



randimg =""

def trainSaveModel():

	print("herer")
	cars = [] # contains file path of cars
	notcars = []
	limit = 300
	for car in glob.glob(datasetPath+"vehicles/vehicles/*/*.png"):
		cars.append(car)
		#if len(cars) > limit : break

	for ncar in glob.glob(datasetPath+"non-vehicles/non-vehicles/*/*.png"):
		notcars.append(ncar)
		#if len(notcars) > limit : break
	global randimg
	randimg = cv2.imread(cars[0])

	print("No of cars: ",len(cars))
	print("No of non-cars: ",len(notcars))
	print("Getting features...")
	carFeatures = [getfeatures(cv2.imread(car)) for car in cars]
	notcarFeatures = [getfeatures(cv2.imread(ncar)) for ncar in notcars]
	print("Got the features...")
	datax = np.vstack((carFeatures, notcarFeatures)).astype(np.float64)
	xscaler = StandardScaler().fit(datax)
	datax = xscaler.transform(datax)
	datay = np.hstack((np.ones(len(carFeatures)), np.zeros(len(notcarFeatures))))
	#print("Carfeatures: ",carFeatures)
	#print("notcarfeatures: ", notcarFeatures)
	#print(datax[0].shape)
	#print(datay)
	print("Training model:")
	trainx, testx, trainy, testy = train_test_split(datax, datay, test_size=0.2)
	svc = svm.LinearSVC()
	start = time.time()
	svc.fit(trainx, trainy)
	end = time.time()
	print("SVC training time: ",end-start) #seconds
	print("Test accuracy: ",svc.score(testx, testy))
	'''
	filename = 'car_classifier_model.sav'
	pickle.dump(svc, open(filename, 'wb'))
	filename = 'xscaler.sav'
	pickle.dump(xscaler, open(filename, 'wb'))
	'''








if __name__ == '__main__':
	#trainSaveModel()
	print("Loading models in main..")
	#svc = pickle.load(open('car_classifier_model.sav', 'rb'))
	#xscaler = pickle.load(open('xscaler.sav', 'rb'))
	#plt.imshow(findcars(randimg), interpolation='nearest')
	#plt.show()

