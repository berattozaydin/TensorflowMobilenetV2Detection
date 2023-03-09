import json

import cv2
import time
import os
import tensorflow as tf
import requests
import numpy as np
classFile = r'C:/Users/berat/PycharmProjects/TensorflowMobilenetV2/coco.names'
classList = []
colorList = []
np.random.seed(123)
with open(classFile,'r') as f:
    classList = f.read().splitlines()
    colorlist = np.random.uniform(low=0,high=255,size=(len(classList),3))

print(len(classList), len(colorList))
#file = tf.keras.utils.get_file(fname='ssd_mobilenet_v3_large_coco_2020_01_14',origin='file://C:/Users/berat/PycharmProjects/TensorflowMobilenetV2/model.tar.gz')
model =tf.keras.applications.MobileNetV2() #tf.saved_model.load(r'ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8/saved_model')
def newModelInput(image):
    frame = image
    image = cv2.resize(image, (224, 224))
    inputs = tf.keras.Input((224,224,3))
    inputTensor = cv2.cvtColor(frame.copy(),cv2.COLOR_BGR2RGB)
    inputTensor = tf.convert_to_tensor(inputTensor,dtype=tf.uint8)
    inputTensor = inputTensor[tf.newaxis,...]
    model = tf.keras.applications.MobileNetV2()
    model.trainable=False

    x = model.output
    fc = tf.keras.layers.Dense(256,activation='relu')(x)
    drop = tf.keras.layers.Dense(0.5)(fc)
    fc2 = tf.keras.layers.Dense(4,activation='sigmoid')(drop)

    newModel = tf.keras.Model(inputs=inputs, outputs=fc2)
    newModel.compile()
    img = image.reshape(1,224,224,3)
    predict = newModel.predict(img)

    return predict

def apiRequest(countPerson):
    try:
        url = f"http://localhost:5155/api/PythonDetection/Write"
        json_data = {
            'PERSON_COUNT': countPerson}
        print(json_data)
        t = json.dumps(json_data)
        x = requests.post(url, json=json_data)
    except Exception as e:
        print(e)
def bboxCoord(image):
    classIndexPerson =0
    countPerson=0
    inputTensor = cv2.cvtColor(image.copy(),cv2.COLOR_BGR2RGB)
    inputTensor = tf.convert_to_tensor(inputTensor,dtype=tf.uint8)
    inputTensor = inputTensor[tf.newaxis,...]
    detect = model(inputTensor)
    bbox = detect['detection_boxes'][0].numpy()
    classIndex = detect['detection_classes'][0].numpy().astype(np.int32)
    classScores = detect['detection_scores'][0].numpy()
    hIm,wIm,cIm = image.shape
    bboxId = tf.image.non_max_suppression(bbox,classScores,max_output_size=50,iou_threshold=0.5,score_threshold=0.5)
    if len(bboxId) !=0:
        for i in bboxId:
            bboxs = tuple(bbox[i].tolist())
            classConf = round(100*classScores[i])
            classIndexes = classIndex[i]
            classLabel = classList[classIndexes-1]
            #classColor = colorList[classIndex]
            if (classList[classIndexes-1] =='person'):
                countPerson=countPerson+1

                classIndexPerson = classList[classIndexes-1]
            text =  '{}: {}%'.format(classLabel,classConf)
            ymin,xmin,ymax,xmax = bboxs
            xmin,xmax,ymin,ymax = (xmin*wIm,xmax*wIm,ymin*hIm,ymax*hIm)
            xmin,xmax,ymin,ymax = int(xmin),int(xmax),int(ymin),int(ymax)
            cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(255,0,0),thickness=1)
            cv2.putText(image,text,(xmin,ymin-10),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)
            print(ymin,xmin,ymax,xmax)
    apiRequest(countPerson)
    return image

camera = cv2.VideoCapture(0)
s=True
while s==True:
   ret,realFrame = camera.read()
   frame=realFrame
   #frame = bboxCoord(realFrame)


   newModel = newModelInput(frame)
   predict = newModel
   print(predict)
   cv2.imshow("Camera Screen", realFrame)
   k = cv2.waitKey(5) & 0xFF
   if k == 27:
       break
camera.release()
cv2.destroyAllWindows()
