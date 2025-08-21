# your_app_name/urls.py
from django.urls import path
from .views import VehicleDamageDetection

urlpatterns = [
    path('detect/', VehicleDamageDetection.as_view(), name='vehicle-damage-detection'),
]
