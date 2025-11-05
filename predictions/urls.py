from django.urls import path
from . import views

app_name = 'predictions'

urlpatterns = [
    path('', views.homepage, name='homepage'),
    path('league/<slug:slug>/', views.league_detail, name='league_detail'),
    path('api/predict/', views.predict_match, name='predict_match'),
]

