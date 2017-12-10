import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
from moviepy.editor import VideoFileClip as vfc

import sys


xscaler = 0 # will be initialized later
svc = 0 # will be initialized later while loading from pickle file
from vehicleDetection import *
################################################
objPoints=[]
imgPoints=[]
#9 corners in row and x corners in y directions
cbrows=9
cbcols=6
objp = np.zeros((cbrows*cbcols,3), np.float32)
objp[:,:2] = np.mgrid[0:cbcols,0:cbrows].T.reshape(-1,2)
#Assume lane width is 3.7 metres
xmpix=3.7/550
ympix=24/720
prev_lx=0
prev_ly=0
prev_rx=0
prev_ry=0

#This function transforms an image into a bird's eye view of he same
def perspective_transform(img):
#    cv2.imshow('FRAME', img)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows() 
    #Four points assuming lane starting and ending points in original    
    src=np.float32([[210,700],[570,460],[705,460],[1075,700]])
    #corresponding points required in output image
    dst=np.float32([[400,720],[400,0],[img.shape[1]-400,0],[img.shape[1]-400,720]])

    #now change the transformation by solving the equations
    M=cv2.getPerspectiveTransform(src,dst)
    MInverse=cv2.getPerspectiveTransform(dst, src)
    #Apply the transformation
    warped_img=cv2.warpPerspective(img,M,(img.shape[1], img.shape[0]))
    unwarped_img=cv2.warpPerspective(warped_img,MInverse,(warped_img.shape[1], img.shape[0]))
    return warped_img,unwarped_img,M,MInverse


def sobel_func(img,min_threshold,max_threshold):
    #Convert to grayscale
    '''Experiment TODO: Check with R Channel/G Channel'''
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    #Calculate in both x and y 
    sobel_img=np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 1))
    #normalize
    sobel_normalized = np.uint8(255*sobel_img/np.max(sobel_img))

    #Create a binary mask and give the final output with thresholding
    output = np.zeros_like(sobel_normalized)
    output[(sobel_normalized>=min_threshold)&(sobel_normalized<=max_threshold)] = 1
    return output



def rChannel_func(img,min_threshold,max_threshold):
    red = img[:,:,0]
    output = np.zeros_like(red)
    output[(red>=min_threshold)&(red<=max_threshold)] = 1
    return output
    
def sChannel_func(img,min_threshold,max_threshold):
    hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    sChannel = hls_img[:,:,2]
    output = np.zeros_like(sChannel)
    output[(sChannel>=min_threshold)&(sChannel<=max_threshold)] = 1
    return output

def lChannel_func(img,min_threshold,max_threshold):
    lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lChannel = lab_img[:,:,2]
    output = np.zeros_like(lChannel)
    output[(lChannel>=min_threshold)&(lChannel<=max_threshold)] = 1
    return output


def combine_threshold(img):
    edge_sobel_img=sobel_func(img,180,255)
    RChannel_img=rChannel_func(img,180,255)
    sChannel_img=sChannel_func(img,180,255)
    lChannel_img=lChannel_func(img,180,255)


    #Now combine all the three channels
    final_binary_img = np.zeros_like(RChannel_img)
    #print(np.logical_or(edge_sobel_img==1, RChannel_img==1, sChannel_img==1))
    final_binary_img[np.logical_or(edge_sobel_img==1, RChannel_img==1, sChannel_img==1)] = 1
    return final_binary_img





def fit_curve1(img):
 #   cv2.imshow('image',img)
    #Debugging output
    global prev_lx
    global prev_ly
    global prev_rx
    global prev_ry
    out_img = np.dstack((img, img, img))*255
    
    left_lane_points=[]
    right_lane_points=[]
    
    '''Take a histogram of the bottom part of the image. wherever the histogram peaks, you got a line'''
    hist=np.sum(img[img.shape[0]//2:,:], axis=0)

    #Take only 1/4th to 3/4th of the image to find your lanes
    midpoint=np.int(hist.shape[0]/2)
    left=np.argmax(hist[0:midpoint])
    right=np.argmax(hist[midpoint:midpoint+midpoint//2])+midpoint
    left_cur=left
    right_cur=right
    #Fix the number of your sliding windows
    windows=9
    #window width in pixels
    sThreshold=100
    #Set min pixels to recenter window
    rThreshold=50
    window_height=np.int(img.shape[0]/windows)
    nz=img.nonzero()
    nzx=np.array(nz[1])
    nzy=np.array(nz[0])
    for win in range(windows):
        xl_low=left_cur-sThreshold
        xl_high=left_cur+sThreshold
        xr_low=right_cur-sThreshold
        xr_high=right_cur+sThreshold
        y_low=img.shape[0] - (win+1)*window_height
        y_high=img.shape[0] - (win)*window_height

        #Draw the windows for debugging
        #cv2.rectangle(out_img,(xl_low,y_low),(xl_high,y_high),(0,255,0),2)
        #cv2.rectangle(out_img,(xr_low,y_low),(xr_high,y_high),(0,255,0),2)
        
        #Now find all the pixels whose values=1 in the windows
        left_pixels=((nzy >= y_low)&(nzy <= y_high)&(nzx >= xl_low)&(nzx <= xl_high)).nonzero()[0]
        right_pixels=((nzy >= y_low)&(nzy <= y_high)&(nzx >= xr_low)&(nzx <= xr_high)).nonzero()[0]
        left_lane_points.append(left_pixels)
        right_lane_points.append(right_pixels)

        if(len(left_pixels) > rThreshold):
            #recenter
            left_cur=np.int(np.mean(nzx[left_pixels]))
        if(len(right_pixels) > rThreshold):
            right_cur=np.int(np.mean(nzx[right_pixels]))
    #Flatten the array
    left_lane_points=np.concatenate(left_lane_points)
    right_lane_points=np.concatenate(right_lane_points)

    lx=nzx[left_lane_points]
    ly=nzy[left_lane_points]
    rx=nzx[right_lane_points]
    ry=nzy[right_lane_points]
    if(len(lx)==0 or len(ly)==0 or len(rx)==0 or len(ry)==0):
        lx=prev_lx
        ly=prev_ly
        rx=prev_rx
        ry=prev_ry
    
    
    lfitpixels=np.polyfit(ly,lx,2)
    rfitpixels=np.polyfit(ry,rx,2)
    lfitmetres=np.polyfit(ly*ympix,lx*xmpix,2)
    rfitmetres=np.polyfit(ry*ympix,rx*xmpix,2)
    prev_lx=lx
    prev_ly=ly
    prev_rx=rx
    prev_ry=ry
    return lfitpixels,rfitpixels,lfitmetres,rfitmetres

def get_vehicle_pos(img,lcurve,rcurve):
    y=img.shape[0]-1
    xl=lcurve[0]*(y**2) + lcurve[1]*(y) + lcurve[2]
    xr=rcurve[0]*(y**2) + rcurve[1]*(y) + rcurve[2]
    result=xmpix*(img.shape[1]/2 - (xl+xr)/2)
    return result

#Final visualization
def visualize(img,bin_img,lfit,rfit,MInverse,lcurv,rcurv,offset):
    
    warp_zero = np.zeros_like(bin_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Generate x,y for plotting
    ploty = np.linspace(0, bin_img.shape[0]-1, bin_img.shape[0] )
    lfitx = lfit[0]*ploty**2 + lfit[1]*ploty + lfit[2]
    rfitx = rfit[0]*ploty**2 + rfit[1]*ploty + rfit[2]
    pts_left = np.array([np.transpose(np.vstack([lfitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([rfitx, ploty])))])

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([np.hstack((pts_left, pts_right))]), (0,255, 255))
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255,0,255), thickness=20)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0,255,255), thickness=20)
    newwarp = cv2.warpPerspective(color_warp,MInverse,(color_warp.shape[1], color_warp.shape[0]))
    
    result = cv2.addWeighted(img, 1, newwarp, 0.5, 0)

    return result

        

def calc_curvature(lcurvemetres,rcurvemetres):    
    y_eval=720. * ympix
    lradcurvature=((1 + (2*lcurvemetres[0]*y_eval + lcurvemetres[1])**2)**1.5) / np.absolute(2*lcurvemetres[0])
    rradcurvature=((1 + (2*rcurvemetres[0]*y_eval + rcurvemetres[1])**2)**1.5) / np.absolute(2*rcurvemetres[0])
    return lradcurvature,rradcurvature



def process_image(img):
    '''
    plt.imshow(img, interpolation='nearest')
    plt.show()
    cimg,found=findcars(img)
    if found:
        plt.imshow(cimg, interpolation='nearest')
        plt.show()
        sys.exit(0)
    '''
    #Get the height and width of the image
    h,  w = img.shape[:2]

    #undistort the image
    undis_img = cv2.undistort(img, mtx, dist, None, mtx)
    

    #Now we need a bird's eye view of the image
    warped_img,unwarped_img,M,MInverse=perspective_transform(undis_img)
    
    #cv2.imshow('FRAME', warped_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows() 

    '''Now we need a robust representation of the lane lines in the image.
    To do that we combine edge detection with hls and hsv thresholding'''
    final_img = combine_threshold(warped_img)

    #Now we need to fit a curve to the lines
    lfitpixels,rfitpixels,lfitmetres,rfitmetres=fit_curve1(final_img)

    lcurvaturemetres,rcurvaturemetres=calc_curvature(lfitmetres,rfitmetres)

    vehicle_position=get_vehicle_pos(undis_img,lfitpixels,rfitpixels)

    result_img=visualize(undis_img,final_img,lfitpixels,rfitpixels,MInverse,lcurvaturemetres,rcurvaturemetres,vehicle_position)
    return findcars(result_img)[0]





#Given a video, process it and save output
def process_video(inp,out):

    vid = vfc(inp)
    prscd_vid=vid.fl_image(process_image)
    prscd_vid.write_videofile(out, audio=False)

#This is to caliberate the camera and remove the distortion
def caliberate_camera(path):
    #Make them global as we need them below
    global ret,mtx,dist,rvecs,tvecs

    for fname in glob.glob(path+'/*'):
        #Take a chessboard image
        img = cv2.imread(fname)
        #Convert to grayscale
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #Find the chessboard corners in the image
        ret, corners = cv2.findChessboardCorners(gray, (cbrows,cbcols),None)
        #If you manage to find them
        if(ret==True):
            #termination criteria epochs and iterations
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray,corners,(15,15),(-1,-1),criteria)
            imgPoints.append(corners2)
            objPoints.append(objp)
            img = cv2.drawChessboardCorners(img, (cbrows,cbcols), corners2,ret)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, gray.shape[::-1],None,None)
if __name__ == '__main__':
    
    
    #First caliberate your camera to get the coefficients to undistort
    caliberate_camera('./camera_cal')

    #Read your video
    process_video('project_video.mp4', 'out.mp4')

    #Test image
    #img = cv2.imread('left12.jpg')
    #h,  w = img.shape[:2]
    #dst = cv2.undistort(img, mtx, dist, None, mtx)

