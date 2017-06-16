# -*- coding: utf-8 -*-
"""
@author: William.Chen
"""

import cv2

def KazeMatch(img1_Path, img2_path):

    # load the image and convert it to grayscale
    img1 = cv2.imread(img1_Path)
    img2 = cv2.imread(img2_path)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # initialize the AKAZE descriptor, then detect keypoints and extract
    # local invariant descriptors from the image
    detector = cv2.AKAZE_create()
    (kps1, descs1) = detector.detectAndCompute(gray1, None)
    (kps2, descs2) = detector.detectAndCompute(gray2, None)

    print("keypoints: {}, descriptors: {}".format(len(kps1), descs1.shape))
    print("keypoints: {}, descriptors: {}".format(len(kps2), descs2.shape))

    # Match the features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(descs1, descs2, k = 2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
            print("m:", m.distance, m.queryIdx, m.trainIdx, m.imgIdx)
            print("n:", n.distance, n.queryIdx, n.trainIdx, n.imgIdx)
            
    # cv2.drawMatchesKnn expects list of lists as matches.
    result = cv2.drawMatchesKnn(img1, kps1, img2, kps2, good, None, flags = 2)

    return result

if __name__ == '__main__':
    result = KazeMatch('python_book.jpg', 'python_book_training2.jpg')
    cv2.imshow("AKAZE matching", result)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()