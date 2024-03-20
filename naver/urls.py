from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.disaster),
    path('naver/', views.search_panorama),
    path('naver/disaster', views.disaster_img)
]