# api/views.py

import torch
from PIL import Image
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from .serializers import ImageUploadSerializer
import json
import os

class VehicleDamageDetection(APIView):
    def post(self, request, format=None):
        serializer = ImageUploadSerializer(data=request.data)
        if serializer.is_valid():
            image = serializer.validated_data['image']
            image_path = f'media/{image.name}'
            
            # Save the uploaded image temporarily
            with open(image_path, 'wb+') as f:
                for chunk in image.chunks():
                    f.write(chunk)

            # Load the pre-trained model
            model_path = 'C:/Users/ALI/Desktop/Projets/Stages/PFE Addinn/vehicle damage detection model/code model/1st_train/weights/best.pt'
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

            # Load and preprocess the image
            img = Image.open(image_path)

            # Perform detection
            results = model(img)
            detections = results.pandas().xyxy[0]

            # Process results
            pieces_dict = {}
            for _, row in detections.iterrows():
                class_name = row['name']
                pieces_dict[class_name] = pieces_dict.get(class_name, 0) + 1

            # Clean up: Remove the uploaded image
            os.remove(image_path)

            # Return results as JSON
            return Response(pieces_dict, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
