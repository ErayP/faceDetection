import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("video2.mp4")

mpFacedetection = mp.solutions.face_detection
faceDetection = mpFacedetection.FaceDetection()
cTime=0
pTime=0

mpDraw = mp.solutions.drawing_utils

while True:
    success , img = cap.read()
    if not success:
        break
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
    result = faceDetection.process(imgRGB)
    
    #print(result.detections)
    
    if result.detections:
        for id ,detection in enumerate(result.detections):
            
            bboxC = detection.location_data.relative_bounding_box
            
            #print(bboxC)
            
            h, w, _ = img.shape
            bbox = int(bboxC.xmin*w),int(bboxC.ymin*h),int(bboxC.width*w),int(bboxC.height*h)
            cv2.rectangle(img, bbox,(255,0,0),2)
    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, "Fps :" + str(int(fps)), (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255))
    
    
    cv2.imshow("img",img)
    k = cv2.waitKey(17) & 0xFF
    if k == 27:
        cap.release()
        cv2.destroyAllWindows()
