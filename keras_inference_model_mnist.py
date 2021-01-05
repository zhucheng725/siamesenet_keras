
from keras.models import load_model
import numpy as np
import random
import keras
from keras import backend as K
import cv2
import time
import os

def contrastive_loss(y_true, y_pred): 
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


    
def get_imgpath():
    path = os.getcwd() 
    if os.path.exists(path + '/result'):
        pass
    else:
        os.mkdir(path + '/result')




def inference_model():

    model_path ='./siamesenet.h5'

    model = load_model(model_path, custom_objects = {'contrastive_loss':contrastive_loss})

    get_imgpath()


    img_1 = cv2.imread('./1_crop_small.jpg',0)
    ret1,img_1=cv2.threshold(img_1,150,255,cv2.THRESH_BINARY)

    img_1 = img_1 / 255.0
    img_1 = cv2.resize(img_1, (28, 28), interpolation=cv2.INTER_CUBIC)
    img_1 = img_1.reshape(1,28,28,1)


    img_2 = cv2.imread('./2_crop_small.jpg',0)
    ret1,img_2=cv2.threshold(img_2,150,255,cv2.THRESH_BINARY)

    img_2 = img_2 / 255.0
    img_2 = cv2.resize(img_2, (28, 28), interpolation=cv2.INTER_CUBIC)
    img_2 = img_2.reshape(1,28,28,1)


    img_3 = cv2.imread('./3_crop_small.jpg',0)
    ret1,img_3=cv2.threshold(img_3,150,255,cv2.THRESH_BINARY)

    img_3 = img_3 / 255.0
    img_3 = cv2.resize(img_3, (28, 28), interpolation=cv2.INTER_CUBIC)
    img_3 = img_3.reshape(1,28,28,1)


    img_4 = cv2.imread('./4_crop_small.jpg',0)
    ret1,img_4=cv2.threshold(img_4,150,255,cv2.THRESH_BINARY)

    img_4 = img_4 / 255.0
    img_4 = cv2.resize(img_4, (28, 28), interpolation=cv2.INTER_CUBIC)
    img_4 = img_4.reshape(1,28,28,1)



    img_5 = cv2.imread('./5_crop_small.jpg',0)
    ret1,img_5=cv2.threshold(img_5,150,255,cv2.THRESH_BINARY)

    img_5 = img_5 / 255.0
    img_5 = cv2.resize(img_5, (28, 28), interpolation=cv2.INTER_CUBIC)
    img_5 = img_5.reshape(1,28,28,1)




    for i in range(500):
        img_path = './photo/' + str(i) + '.jpg'
        img =  cv2.imread(img_path, 1)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        #img = img[366:400,1445:1469]
        img = img[162:177,480:490]
        ret1,img=cv2.threshold(img,150,255,cv2.THRESH_BINARY)
  
    
        img = img / 255.0
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)
        img = img.reshape(1,28,28,1)

        vs_1 = model.predict([img, img_1])
        vs_2 = model.predict([img, img_2])
        vs_3 = model.predict([img, img_3])
        vs_4 = model.predict([img, img_4])
        vs_5 = model.predict([img, img_5])


        result = np.argmin(np.array([vs_1[0][0],vs_2[0][0],vs_3[0][0],vs_4[0][0],vs_5[0][0]]),0)
        if int(result) == 0:
            reg = 1
        elif int(result) == 1:
            reg = 2
        elif int(result) == 2:
            reg = 3
        elif int(result) == 3:
            reg = 4
        elif int(result) == 4:
            reg = 5

        img = cv2.imread(img_path,1)
        write_path = './result/frame_' + str(i) + '.jpg'
        text = 'FLOOR: '+str(reg)
        cv2.putText(img, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (142,252,0), 1)
        cv2.rectangle(img, (480, 162), (490, 177), (142,252,0), 1)  
        cv2.imwrite(write_path,img)

        print('result:'+str(i),reg)




def main():
    inference_model()

if __name__ == '__main__':
    main()

