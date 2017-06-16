# -*- coding: utf-8 -*-
"""
@author: William.Chen
"""
import glob
import cv2
import numpy as np

MIN_MATCH_COUNT = 10

# color : blue、greem、red、Fuchsia、yellow
color = [(255,0, 0),(0,255,0),(0,0,255),(255, 0,255),(0,255,255)]

def loadImages(path):
    filenames = [img for img in glob.glob(path)]

    filenames.sort() # ADD THIS LINE

    images = []
    for img in filenames:
        n = cv2.imread(img)
        images.append(n)

    return images
if __name__ == '__main__':    
    # Load images and convert to gray level
    modelsBGR = loadImages("targets/*.png")
    modelsGray = [cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY) for gray in modelsBGR]
    
    ## Initiate AKAZE detector and BruteForce Matcher
    akaze = cv2.AKAZE_create()
    bf = cv2.BFMatcher()
    
    # find the key points with AKAZE for the Template Image
    kpsModels, descsModels = ([] for i in range(2))

    for gray in modelsGray:
        (kps, descs) = akaze.detectAndCompute(gray, None)
        kpsModels.append(kps)
        descsModels.append(descs)
        
    # add a border to every templete's image
    for i in range(len(modelsBGR)):
        modelsBGR[i]=cv2.copyMakeBorder(modelsBGR[i], top=10, bottom=10, left=10, right=10, borderType= cv2.BORDER_ISOLATED, value = list(color[i]) )
    # Load video
    video = cv2.VideoCapture("Test Video.avi")

    # Get video size
    vHeight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vWidth = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

    #Process for thumbnail, get ratio and concat images together
    thumbnailHeight =  vHeight / len(modelsBGR)
    print(thumbnailHeight)
    height, width = max(img.shape[:2] for img in modelsBGR)
    print(height, width)
    subsamplingRatio = thumbnailHeight / height
    thumbnailWdith = subsamplingRatio * width
    print(thumbnailHeight, thumbnailWdith)
    thumbnails = [cv2.resize(img,(0, 0), fx = subsamplingRatio, fy = subsamplingRatio) for img in modelsBGR]
    

    models = cv2.vconcat(thumbnails)
    print(thumbnails[0].shape)
    # Create result image 
    rHeight = vHeight
    rWidth = int(thumbnailWdith) + vWidth
    result = np.zeros((rHeight, rWidth, 3), np.uint8)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    writer = cv2.VideoWriter('Result.avi', fourcc, 20.0, (rWidth, rHeight))
    print((vHeight, vWidth))
    print("result",(rHeight, rWidth))
while(video.isOpened()):
     # Capture frame-by-frame
     retval, frame = video.read()
     if retval is not True:
        break

     # Pre-process the Video frame and get gray Scale
     frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

     # find the key points with AKAZE For the Camera Image
     kpFrame, desFrame = akaze.detectAndCompute(frameGray, None)
     # find the Matches between video and images of points.
     matches = [bf.knnMatch(des, desFrame, k = 2) for des in descsModels]

     # Apply ratio test
     ratio = 0.7
     good = [[] for i in range(len(modelsBGR))]
     for x in range(len(modelsBGR)):
         for m, n in  matches[x]:
             if m.distance < ratio * n.distance:
                 good[x].append(m)

     result = cv2.hconcat((models, frame))
     for i in range(len(modelsBGR)):
         if len(good[i]) > MIN_MATCH_COUNT: # Homography
            src_pts = np.float32([ kpsModels[i][m.queryIdx].pt for m in good[i] ]).reshape(-1, 1, 2)
            dst_pts = np.float32([ kpFrame[m.trainIdx].pt for m in good[i] ]).reshape(-1, 1, 2)
            
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            h, w = modelsBGR[i].shape[:2]
            pts = np.float32([ [0, 0], [0, h-1], [w-1, h-1], [w-1, 0] ]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            
            for idx in range(len(dst)):
                dst[idx][0][0] += thumbnailWdith

            result = cv2.polylines(result,[np.int32(dst)], True, color[i], 3, cv2.LINE_AA)


            for idx in range(len(mask)):
                if mask[idx]:
                    src_pts[idx][0][0] *= subsamplingRatio
                    src_pts[idx][0][1]  = src_pts[idx][0][1] * subsamplingRatio +  thumbnailHeight * i
                    dst_pts[idx][0][0] += thumbnailWdith
                    cv2.line(result, tuple(src_pts[idx][0]), tuple(dst_pts[idx][0]), color[i], 1)


         else:
            # print ("Not enough matches are found - %d/%d" % (len(good[i]),MIN_MATCH_COUNT))
            matchesMask = None

     # writer.write(result)
     cv2.imshow("frame", result)
     if cv2.waitKey(1) & 0xFF == ord('q'):
         break

# When everything done, release the video
video.release()
writer.release()
cv2.destroyAllWindows()