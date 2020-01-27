import tensorflow.keras
from PIL import Image
import cv2
import numpy as np

cap = cv2.VideoCapture(1)

model = tensorflow.keras.models.load_model('SetSolverModel_v3_final.h5')
CLASS_NAMES = ['twoMediumPurpleOval', 'twoFullRedWave','twoMediumRedOval',
 'twoMediumGreenWave', 'twoMediumPurpleWave', 'twoMediumGreenDiamond',
 'twoMediumRedDiamond', 'twoMediumGreenOval', 'twoMediumPurpleDiamond',
 'twoMediumRedWave', 'twoFullRedOval', 'twoEmptyRedOval' ,'twoFullRedDiamond',
 'twoFullGreenOval', 'twoFullPurpleDiamond', 'twoFullGreenWave',
 'twoFullGreenDiamond', 'twoFullPurpleOval' ,'twoEmptyRedWave',
 'twoFullPurpleWave', 'twoEmptyGreenWave' ,'twoEmptyGreenOval',
 'threeMediumRedWave', 'threeMediumRedDiamond' ,'twoEmptyGreenDiamond',
 'twoEmptyPurpleOval', 'threeMediumRedOval', 'twoEmptyPurpleDiamond',
 'twoEmptyRedDiamond', 'twoEmptyPurpleWave', 'threeFullRedDiamond',
 'threeMediumGreenOval', 'threeFullPurpleWave', 'threeMediumGreenWave',
 'threeMediumPurpleOval', 'threeFullRedOval' ,'threeFullRedWave',
 'threeMediumPurpleWave', 'threeMediumGreenDiamond',
 'threeMediumPurpleDiamond', 'threeFullGreenDiamond' ,'threeEmptyPurpleOval',
 'threeEmptyRedDiamon', 'threeEmptyPurpleWave' ,'threeEmptyRedWave',
 'threeFullPurpleDiamond', 'threefullGreenWave' ,'threeFullPurpleOval',
 'threeEmptyRedOval', 'threeFullGreenOval', 'threeEmptyGreenDiamond',
 'threeEmptyPurpleDiamond', 'oneMediumPurpleOval', 'oneMediumRedDiamond',
 'threeEmptyGreenWave', 'oneMediumPurpleDiamond', 'oneMediumRedWave',
 'oneMediumRedOval', 'threeEmptyGreenOval', 'oneMediumPurpleWave',
 'oneMediumGreenOval', 'oneMediumGreenWave' ,'oneFullPurpleDiamond',
 'oneFullRedDiamond', 'oneFullGreenWave' ,'oneFullPurpleOval',
 'oneFullPurpleWave', 'oneFullRedWave', 'oneMediumGreenDiamond',
 'oneFullRedOval', 'oneEmptyPurpleDiamond', 'oneEmptyRedOval',
 'oneEmptyRedWave', 'oneEmptyPurpleWave', 'oneFullGreenDiamond',
 'oneEmptyGreenOval', 'oneEmptyPurpleOval', 'oneFullGreenOval',
 'oneEmptyRedDiamond' ,'oneEmptyGreenWave', 'oneEmptyGreenDiamond']



while True:
  _, frame = cap.read()
  bluerred_frame = cv2.GaussianBlur(frame,(5,5),0)
  hsv = cv2.cvtColor(bluerred_frame, cv2.COLOR_BGR2HSV)
  
  


  # whole card
  sensitivity = 80
  lower_white = np.array([0,0,255-sensitivity])
  upper_white = np.array([255,sensitivity,255])

  mask = cv2.inRange(hsv, lower_white, upper_white)
  # Bitwise-AND mask and original image
  res = cv2.bitwise_and(frame,frame, mask= mask)

  ret,thresh = cv2.threshold(mask, 40, 255, 0)
  # Contours 
  contours, _= cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
  # print("the contours are : " , contours)


  data = np.ndarray(shape=(1, 160, 160, 3), dtype=np.float32)

  n=0
  if len(contours) !=0:
      
      for contour in contours:

        area = cv2.contourArea(contour)
        
        
        if area > 10000:

          n=n+1
          x,y,w,h = cv2.boundingRect(contour)
          cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,0),2) 
          # Extract the frame of each card 
          roi = frame[y:y+h,x:x+w]
          
          # Save cards 
          cv2.imwrite(f"cardsE/roi{n}.jpg", roi)
          # print("hmm:", n)
          image = Image.open(f'cardsE/roi{n}.jpg')
          image = image.resize((160, 160))
          image_array = np.asarray(image)

          # # Normalize the image
          # normalized_image_array = image_array.astype(np.float32) / 255.0 
          normalized_image_array = (image_array.astype(np.float32) / 400.0) 

          # Load the image into the array
          data[0] = normalized_image_array

          # run the inference
          prediction = model.predict(data)

          # print(f'Our Model Predicttion : {prediction}')
          
          pred_id=np.argmax(prediction,axis=-1)
          # print(pred_id)
          pred_label=CLASS_NAMES[int(pred_id)]
          PC=prediction[0][pred_id]*100
          print(f'The result is : {pred_label} with {float(PC)} Accuracy %')
          name= str(pred_label) + ":" + str(PC)
          cv2.putText(frame,name,(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),0)
          

  # data = np.ndarray(shape=(1, 160, 160, 3), dtype=np.float32)

  # for i in range(n):
	 #  image = Image.open(f'cardsE/roi{i+1}.jpg')
	 #  image = image.resize((160, 160))
	 #  image_array = np.asarray(image)

	 #  # # Normalize the image
	 #  # normalized_image_array = image_array.astype(np.float32) / 255.0 
	 #  normalized_image_array = (image_array.astype(np.float32) / 400.0) 

	 #  # Load the image into the array
	 #  data[0] = normalized_image_array

	 #  # run the inference
	 #  prediction = model.predict(data)

	 #  # print(f'Our Model Predicttion : {prediction}')
	  
	 #  pred_id=np.argmax(prediction,axis=-1)
	 #  # print(pred_id)
	 #  pred_label=CLASS_NAMES[int(pred_id)]
	 #  PC=prediction[0][pred_id]*100
	 #  print(f'The result is : {pred_label} with {float(PC)} Accuracy %')



  cv2.imshow('frame',frame)
  # cv2.imshow('mask',mask)
  # cv2.imshow('res',res)

  key = cv2.waitKey(10)

  if key == 27:
    break

cap.release()
cv2.destroyAllWindows()



