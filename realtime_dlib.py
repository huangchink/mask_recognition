import numpy as np
import os,cv2,glob
from  tensorflow.python.keras.utils import np_utils
import matplotlib.pyplot as plt
from  tensorflow.python.keras.models import Sequential
from  tensorflow.python.keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten,Dense
from  tensorflow.python.keras.models import load_model
import dlib
import imutils
import time
np.random.seed(10)
dict_labels = ["mask","nomask"]

def show_images_labels_predictions(images,labels,
                                  predictions,start_id,num=25):
    plt.gcf().set_size_inches(12, 14)
    if num>30: num=30 
    for i in range(0, num):
        ax=plt.subplot(5,5, 1+i)
        #顯示彩色圖片
        ax.imshow(images[start_id])
        
        # 有 AI 預測結果資料, 才在標題顯示預測結果
        if( len(predictions) > 0 ) :
            title = 'ai=>' + dict_labels[predictions[start_id]]
            title +=('(pass)' if predictions[start_id]==labels[start_id] else '(fail)')
            # 預測正確顯示(pass), 錯誤顯示(fail)
    
           # title += '\nlabel = '+dict_labels[start_id]
        # 沒有 AI 預測結果資料, 只在標題顯示真實數值
        else :
            title = 'label = ' + dict_labels[start_id]
            
        # X, Y 軸不顯示刻度    
        ax.set_title(title,fontsize=12) 
        
        ax.set_xticks([]);ax.set_yticks([])        
   
        start_id+=1 
    plt.show()

imagesavepath='npy_Dataset/'

try: 
    print("載入模型 cnn_model.h5")
    model = load_model('cnn_model.h5') 
except:
    for i in range(1):
        train_feature=np.load(imagesavepath+'train_feature.npy')  
        test_feature=np.load(imagesavepath+'test_feature.npy')  
        train_label=np.load(imagesavepath+'train_label.npy')      
        test_label=np.load(imagesavepath+'test_label.npy')       
        print("載入 *.npy 檔!") 
   
    # 將 Features 特徵值換為 圖片數量*80*80*3 的 4 維矩陣
        train_feature_vector =train_feature.reshape(len(train_feature), 180,180,3).astype('float32')
        test_feature_vector = test_feature.reshape(len( test_feature), 180,180,3).astype('float32')
    
    #Features 特徵值標準化
        train_feature_normalize = train_feature_vector/255
        test_feature_normalize = test_feature_vector/255
    
    #label 轉換為 One-Hot Encoding 編碼
        train_label_onehot = np_utils.to_categorical(train_label)
        test_label_onehot = np_utils.to_categorical(test_label)
    
    #建立模型
        model = Sequential()
    #建立卷積層1
        model.add(Conv2D(filters=10, 
                      kernel_size=(5,5),
                      padding='same',
                      input_shape=(180,180,3), 
                      activation='relu'))
    
    #建立池化層1
        model.add(MaxPooling2D(pool_size=(2, 2))) #(10,40,40)
    
    # Dropout層防止過度擬合，斷開比例:0.1
        model.add(Dropout(0.1))    
    
    #建立卷積層2 #(20,40,40)
        model.add(Conv2D(filters=20, 
                      kernel_size=(5,5),  
                      padding='same',
                      activation='relu'))
    
    #建立池化層2
        model.add(MaxPooling2D(pool_size=(2, 2))) #(20,20,20)
    
    # Dropout層防止過度擬合，斷開比例:0.2
        model.add(Dropout(0.5))
    
    #建立平坦層：20*20*20=8000 個神經元
        model.add(Flatten()) 
    
    #建立隱藏層
        model.add(Dense(units=256, activation='relu'))
    
    #建立輸出層
        model.add(Dense(units=2,activation='softmax'))
    
        model.summary() #顯示模型
    
    #定義訓練方式
        model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', metrics=['accuracy'])
    
    #以(train_feature_normalize,train_label_onehot)資料訓練，
    #訓練資料保留 10%作驗證,訓練30次、每批次讀取10筆資料，顯示簡易訓練過程
        train_history =model.fit(x=train_feature_normalize,
                             y=train_label_onehot,validation_split=0.4, 
                             epochs=20, batch_size=10,verbose=2)
    #評估準確率
    #print(train_label_onehot)
        fuck=model.evaluate(train_feature_normalize,train_label_onehot)
        print('\n train準確率=',fuck[1])
        scores = model.evaluate(test_feature_normalize, test_label_onehot)
        print('\n test準確率=',scores[1])
        
    #預測
        prediction=model.predict_classes(test_feature_normalize)
        model.save('cnn_model.h5')
        print("cnn_model.h5 模型儲存完畢!")
        model.save_weights("cnn_model.weight")
        print("cnn_model.weight 模型參數儲存完畢!")
 
    
    #顯示圖像、預測值、真實值 
        show_images_labels_predictions(test_feature,test_label,prediction,0)
    
   
images=[]
img_label=[]
try:
        
    img=cv2.imread('av.jpeg')
        
    img=cv2.resize(img,dsize=(180,180))
  
    img_label.append(1)
            
    images.append(img)
        
            
    img=cv2.imread('west.jpg')
        
    img=cv2.resize(img,dsize=(180,180))
            
    img_label.append(1)
            
    images.append(img)
       
        
    img=cv2.imread('chink.jpg')
        
    img=cv2.resize(img,dsize=(180,180))
            
    img_label.append(0)
            
    images.append(img)
        
        
    img=cv2.imread('water.jpg')
        
    img=cv2.resize(img,dsize=(180,180))
            
    img_label.append(1)
            
    images.append(img)
       
    img=cv2.imread('eren.jpg')
        
    img=cv2.resize(img,dsize=(180,180))
            
    img_label.append(1)
            
    images.append(img)
        
    img=cv2.imread('blow.jpg')
        
    img=cv2.resize(img,dsize=(180,180))
            
    img_label.append(0)
            
    images.append(img)
        
    img=cv2.imread('up.jpg')
        
    img=cv2.resize(img,dsize=(180,180))
            
    img_label.append(1)
            
    images.append(img)
        
    img=cv2.imread('crayon.jpg')
        
    img=cv2.resize(img,dsize=(180,180))
            
    img_label.append(1)
            
    images.append(img)
        
    img=cv2.imread('james.jpg')
        
    img=cv2.resize(img,dsize=(180,180))
            
    img_label.append(1)
            
    images.append(img)
        
    img=cv2.imread('mora.jpg')
        
    img=cv2.resize(img,dsize=(180,180))
            
    img_label.append(0)
            
    images.append(img)
        
except:
    print("無法讀取!")
    pass
images=np.array(images)

images = images.reshape(len(images),180,180,3).astype('float32')

images = images/255

prediction1=model.predict_classes(images)

    
    
show_images_labels_predictions(images,img_label,prediction1,0,10)
    
    
    
    
cap = cv2.VideoCapture(0)

# Dlib 的人臉偵測器
detector = dlib.get_frontal_face_detector()
framecount=0
FPS=''
while(cap.isOpened()):
    
        
    
    ret, frame = cap.read()
    
    frame = imutils.resize(frame, width=1024)
  # 偵測人臉
    face_rects, scores, idx = detector.run(frame, 0,-0.2)
    
    t_start = time.time()
    
# 取出所有偵測的結果
    imgg=[]
    
    for i, d in enumerate(face_rects):
        x1 = d.left()
        y1 = d.top()
        x2 = d.right()
        y2 = d.bottom()
  # 以方框標示偵測的人臉
        fake=frame[y1:y2,x1:x2]
        fake=cv2.resize(fake,dsize=(180,180))
        imgg.append(fake)
        imggg=np.array(imgg)
        imggg = imggg.reshape(len(imggg),180,180,3).astype('float32')
        imggg=imggg/255
        guess =model.predict_classes(imggg)
          
        
       
        
                 
        cv2.putText(frame, FPS, (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            
        if guess[i]==0:
            text='mask'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255,0), 2)
            cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_DUPLEX,0.7, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            text='no mask'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_DUPLEX,
                            0.7, (255, 255, 255), 1, cv2.LINE_AA)
        FPS = "FPS=%1f" % (1 / (time.time() - t_start))
        cv2.putText(frame, FPS, (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
        
# 顯示結果
    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
    
    
    
    
    
    
    
    
'''except:
    print(".npy 檔未建立!")    '''