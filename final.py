#PROJECT - VIRTUAL ADVERTISING USING AR

#  <-----------importing modules-------------->
import cv2
import numpy as np 

#<--------------Camera Frame Read----------------->
#cap = cv2.VideoCapture('http://192.168.29.20:8080/video')
cap = cv2.VideoCapture(1)
#--------------Input the image path---------------->
# imgT1 = input("enter the target image path")
# img1O1 = input("enter the out image 1 path")
# img1O2 = input("enter the out image 2 path")

#<-----------------Target Image 1----------------------->
imgTarget = cv2.imread('vimal.jpg')

#<------------------Output images for Target 1---------------->n  
myVid = cv2.VideoCapture('images/Aus1.jpg')
myVid1 = cv2.VideoCapture('images/ENG1.jpg')

detection = False
frameCounter = 0
#<---------------Resize the output image according to the target image---------->
success,imVideo = myVid.read()
hT,wT,cT = imgTarget.shape
imVideo = cv2.resize(imVideo,(wT,hT))

success1,imVideo1 = myVid1.read()
imVideo1 = cv2.resize(imVideo1,(wT,hT))

# imgT2 = input("enter the target image path")
# img2O1 = input("enter the out image 1 path")
# img2O2 = input("enter the out image 2 path")

#<-----------------Target Image 2----------------------->
imgTarget1 = cv2.imread('Tata.jpg')

#<------------------Output images for Target 2---------------->
myVid2 = cv2.VideoCapture('images/Aus2.jpg')
myVid3 = cv2.VideoCapture('images/ENG2.jpg')
# myVid2 = cv2.VideoCapture('')

#<---------------Resize the output image according to the target image---------->
success2,imVideo2 = myVid2.read()
hT1,wT1,cT1 = imgTarget1.shape
imVideo2 = cv2.resize(imVideo2,(wT1,hT1))

success3,imVideo3 = myVid3.read()
imVideo3 = cv2.resize(imVideo3,(wT1,hT1))

#<-------------------Feature Creation for Target images----------------->
orb = cv2.ORB_create(nfeatures=1000)
kp1,des1 = orb.detectAndCompute(imgTarget,None)
imgTarget = cv2.drawKeypoints(imgTarget,kp1,None)

kp3,des3 = orb.detectAndCompute(imgTarget1,None)
imgTarget1 = cv2.drawKeypoints(imgTarget1,kp3,None)

#<-------------------Finale Output image stacking------------------->
def stackImages(imgArray,scale,lables=[]):
    sizeW= imgArray[0][0].shape[1]
    sizeH = imgArray[0][0].shape[0]
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (sizeW, sizeH), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (sizeW, sizeH), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth= int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        print(eachImgHeight)
        for d in range(0, rows):
            for c in range (0,cols):
                cv2.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d][c])*13+27,30+eachImgHeight*d),(255,255,255),cv2.FILLED)
                cv2.putText(ver,lables[d][c],(eachImgWidth*c+10,eachImgHeight*d+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
    return ver

#<---------------Real time tracking and Augmentation---------------->
while True:
    _,out = cap.read()
    #<--------Read the webcam frame------->
    sucess,imgWebcam = cap.read()
    a,finalCap = cap.read()
    #<----initialize aug value as imgAug--------->
    imgAug = imgWebcam.copy()
    imgAug2 = imgWebcam.copy()
    #<---------Detect feature from realtime camera feed--------->
    kp2,des2 = orb.detectAndCompute(imgWebcam,None)
    kp4,des4 = orb.detectAndCompute(imgWebcam,None)
    #<---------Draw Point for the detected Features---------->
    imgWebcam = cv2.drawKeypoints(imgWebcam,kp2,None)
    #<------------KNN matching for target image and camera frame image------>
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    good = []
    for m,n in matches : 
        if(m.distance < 0.75 *n.distance):
            good.append(m)
    print(len(good))
    #<--------Draw the matched features in realtime---------->
    imgFeatures = cv2.drawMatches(imgTarget,kp1,imgWebcam,kp2,good,None,flags=2)
    imgStacked = stackImages(([imgWebcam,imgTarget,imgFeatures],[out,out,out]),0.5)
    # kp4,des4 = orb.detectAndCompute(imgWebcam,None)
    # imgWebcam = cv2.drawKeypoints(imgWebcam,kp4,None)  
    #<---------if the match is more than 20 than invoke augmentation--------->
    if len(good) > 20:
        #<--------draw polylines according to the image match-------->
        srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dstPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        matrix, mask = cv2.findHomography(srcPts,dstPts,cv2.RANSAC,5)
        print(matrix)

        pts = np.float32([[0,0],[0,hT],[wT,hT],[wT,0]]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,matrix)
        img2 = cv2.polylines(imgWebcam,[np.int32(dst)],True,(255,0,255),3)
        #<-------image Warp for the target image -------->
        imgWarp = cv2.warpPerspective(imVideo,matrix, (imgWebcam.shape[1],imgWebcam.shape[0]))
        imgWarp1 = cv2.warpPerspective(imVideo1,matrix, (imgWebcam.shape[1],imgWebcam.shape[0]))
        #<------------- Create mask corresponding to the camera feed--------->
        maskNew = np.zeros((imgWebcam.shape[0],imgWebcam.shape[1]),np.uint8)
        cv2.fillPoly(maskNew,[np.int32(dst)],(255,255,255))
        #<------Inverse the mask--------->
        maskInv = cv2.bitwise_not(maskNew)
        imgAug = cv2.bitwise_and(imgAug,imgAug,mask = maskInv)
        imgAug1 = imgAug
        #<------mask collaboration and augmentation--------->
        imgAug = cv2.bitwise_or(imgWarp,imgAug)
        imgAug1 = cv2.bitwise_or(imgWarp1,imgAug1)

        imgStacked = stackImages(([imgWebcam,imgTarget,imgFeatures],[out,imgAug,imgAug1]),0.5)
    
    imgWebcam = imgAug2.copy()
    imgWebcam = cv2.drawKeypoints(imgWebcam,kp4,None)
    bf1 = cv2.BFMatcher()
    matches1 = bf1.knnMatch(des3,des4,k=2)
    good1 = []
    for m1,n1 in matches1 : 
        if(m1.distance < 0.75 *n1.distance):
            good1.append(m1)
    print(len(good1))
    imgFeatures1 = cv2.drawMatches(imgTarget1,kp3,imgWebcam,kp4,good1,None,flags=2)

    if len(good1) > 20:
        srcPts1 = np.float32([kp3[m1.queryIdx].pt for m1 in good1]).reshape(-1,1,2)
        dstPts1 = np.float32([kp4[m1.trainIdx].pt for m1 in good1]).reshape(-1,1,2)

        matrix1, mask1 = cv2.findHomography(srcPts1,dstPts1,cv2.RANSAC,5)
        print(matrix1)

        pts1 = np.float32([[0,0],[0,hT1],[wT1,hT1],[wT1,0]]).reshape(-1,1,2)
        dst1 = cv2.perspectiveTransform(pts1,matrix1)
        img3 = cv2.polylines(imgWebcam,[np.int32(dst1)],True,(255,0,255),3)
         
        imgWarp2 = cv2.warpPerspective(imVideo2,matrix1, (imgWebcam.shape[1],imgWebcam.shape[0]))
        imgWarp3 = cv2.warpPerspective(imVideo3,matrix1, (imgWebcam.shape[1],imgWebcam.shape[0]))

        maskNew1 = np.zeros((imgWebcam.shape[0],imgWebcam.shape[1]),np.uint8)
        cv2.fillPoly(maskNew1,[np.int32(dst1)],(255,255,255))
        maskInv1 = cv2.bitwise_not(maskNew1)
        imgAug2 = cv2.bitwise_and(imgAug2,imgAug2,mask = maskInv1)
        imgAug3 = imgAug2
        finalImgAug2 = imgAug2
        imgAug2 = cv2.bitwise_or(imgWarp2,imgAug2)
        
        imgAug3 = cv2.bitwise_or(imgWarp3,imgAug3)

        imgStacked = stackImages(([imgWebcam,imgTarget1,imgFeatures1],[out,imgAug2,imgAug3]),0.5)   

    #cv2.imshow('maskNew',imgAug)    
    #cv2.imshow('imgWrap',imgWarp)
    #cv2.imshow('img2',img2)
    #cv2.imshow('imgFeatures',imgFeatures)
    # cv2.imshow('imgFeatures',cv2.resize(imgFeatures,(960,480)))
    #cv2.imshow('imgTarget',imgTarget)
    #cv2.imshow('myVid',imVideo)
    # cv2.imshow('India',imgWebcam)
    # cv2.imshow('Australia',imgAug)
    # cv2.imshow('USA',imgAug1)

    
    #<-----------final output displayed--------->
    try:
        # cv2.imshow('imgStacked',cv2.resize(imgStacked,(960,480)))
        # cv2.imshow('imgStacked',cv2.resize(imgStacked1,(960,480)))
        cv2.imshow('vimal',imgAug1)
        cv2.imshow('tata',imgAug2)
        cv2.imshow('outstream',cv2.resize(imgStacked,(960,480)))
        # finalMask = cv2.bitwise_and(maskInv,maskInv1)
        # finalMask = cv2.bitwise_not(finalMask)
        # finalWrap = cv2.bitwise_and(imgWarp1,imgWarp2)
        # finalAug = cv2.bitwise_and(finalImgAug,finalImgAug2)
        # imgAug = cv2.bitwise_and(imgAug,imgAug,mask = finalMask)
        # final = cv2.bitwise_or(imgWarp,imgAug3)
        # # final = cv2.bitwise_or(imgWarp2,imgAug1)
        # # final = cv2.bitwise_and(final,final,mask=finalMask)
        # cv2.imshow('final',final)
    except:
        pass
    #cv2.resize(imgStacked,(480,960))
    cv2.waitKey(1)