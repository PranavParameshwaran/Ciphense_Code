from flask import Flask, jsonify, request, send_file
import numpy as np
import cv2
import os
from imageai.Detection import ObjectDetection
	
app = Flask(__name__)

@app.route('/',methods=['POST'])
def home():

	filestr = request.files
	image = filestr['img'].read()
	#convert string data to numpy array
	npimg = np.fromstring(image, np.uint8)
	img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
	cv2.imwrite("newImg.jpeg", img)
	
	execution_path = os.getcwd()

	detector = ObjectDetection()
	detector.setModelTypeAsRetinaNet()
	detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
	detector.loadModel()
	detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "newImg.jpeg"), output_image_path=os.path.join(execution_path , "imagenew.jpg"))

	animal_s = ['bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe']
	person =0
	animal =0
	objects =0
	for eachObject in detections:
	    if eachObject["name"]=="person" and eachObject["percentage_probability"] >= 55:
	    	person+=1
	    if eachObject["name"]in animal_s and eachObject["percentage_probability"] >= 55:
	    	animal+=1
	    else:
	    	objects+=1
	Ans = {'Person':person,'Animal':animal, 'object':objects-person}
	return jsonify(Ans)
	

if __name__ == '__main__':
 app.run(debug = True)

