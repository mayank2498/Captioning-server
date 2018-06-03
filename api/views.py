from django.shortcuts import render,HttpResponse
import make_predictions
from make_predictions import predict
from PIL import Image
import numpy as np
import cv2

def index(request):
	if request.method == 'GET':
		print("get req")
		return render(request,'api/index.html')
	

	image = request.FILES['img']
	img = Image.open(image)

	img = cv2.resize(np.array(img).astype('float32'),(224,224))
	print("received Image")

	return HttpResponse(predict(img))
