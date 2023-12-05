from django.urls import path

from main.views import *

app_name = 'main'

urlpatterns = [
    path('', home, name='home'),
    path('search', search, name='search'),
    path('detail/<path:path>', detail, name='detail'),
    path('about', about, name='about'),
]