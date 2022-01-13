from django.urls import path
from . import views

app_name = 'edit'
urlpatterns = [
    path('', views.index, name='index'),
]
