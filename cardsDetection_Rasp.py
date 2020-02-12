import tensorflow as tf
from PIL import Image
import cv2
import numpy as np
import findSet as f
import re
from tflite_runtime.interpreter import load_delegate

cap = cv2.VideoCapture(0)

#model = tensorflow.keras.models.load_model('SetSolverModel_v3_final.h5')
# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="setSolver.tflite",experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
interpreter.allocate_tensors()


# Get input and output tensors.
input_details = interpreter.get_input_details()
#print("in :",input_details)
output_details = interpreter.get_output_details()
#print("out :",output_details)

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
print(" height & width: ",height,width)


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
  #sensitivity = 80
  sensitivity = 70
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
  cards=[]
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
          	image = Image.open(f'cardsE/roi{n}.jpg')
          	image = image.resize((160, 160))
          	image_array = np.asarray(image)

          	# # Normalize the image
          	#normalized_image_array = image_array.astype(np.float32) / 255.0 
          	normalized_image_array = (image_array.astype(np.float32) / 400.0) 

          	# Load the image into the array
          	data = normalized_image_array
				#data[ = normalized_image_array

          	# run the inference
          	#prediction = model.predict(data)
          	input_data = np.expand_dims(data, axis=0)
          	#input_data = np.expand_dims(data[0], axis=0)
          	#print("hmm",len(input_data))
          	#print("hmm: ",input_details[0]['shape'])
          	interpreter.set_tensor(input_details[0]['index'], input_data)
          	interpreter.invoke()
			 	

			 	# The function `get_tensor()` returns a copy of the tensor data.
          	# Use `tensor()` in order to get a pointer to the tensor.
          	#output_data = interpreter.get_tensor(output_details[0]['index'])
          	prediction = interpreter.get_tensor(output_details[0]['index'])
          	#interpreter.invoke()  
          	#print(output_data)        

          	# print(f'Our Model Predicttion : {prediction}')
          
          	pred_id=np.argmax(prediction,axis=-1)
          	#print(pred_id)
          	pred_label=CLASS_NAMES[int(pred_id)]
          
          	#print(pred_label)
          	#Save cards name
          	cards.append(pred_label)
          	#print the results of detection
          	PC=prediction[0][pred_id]*100
          	#print(f'The result is : {pred_label} with {float(PC)} Accuracy %')
          	name= str(pred_label) + ":" + str(PC)
          	 
          	cv2.putText(frame,name,(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),0)
          	 
  				
  #interpreter.invoke()        
  print("cards: ",cards)
  cv2.imshow('frame',frame)
  cv2.imshow('mask',mask)
  
  
  
  
  def extractCards(cards):
    collection=[]   
    for card in cards:
      #Split 'oneFullPurpleOval' to  ['one', 'full', 'purple', 'oval']
      s=[]
      p=re.findall('^[a-z]+|[A-Z][^A-Z]*', card)
      [s.append(i.lower()) if not i.islower() else s.append(i) for i in p]
      k=["id","color","shape","fill","number"]
      # val=["one_blue_empty_diamond","blue","diamonds","empty","one"]
      gId=s[0]+"_"+s[2]+"_"+s[1]+"_"+s[3]
      val=[gId,s[2],s[3],s[1],s[0]]
      
      zipObj = zip(k,val)
      mydict = dict(zipObj)
      collection.append(mydict)

    return collection
  
  collection=extractCards(cards)
  print(f"We found {len(f.getSets(collection))} sets :",f.getSets(collection))
  
  key = cv2.waitKey(1)

  if key == 27:
    break

cap.release()
cv2.destroyAllWindows()



