from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
from .views import *

urlpatterns = [
    path('', views.index, name='index'),
    path('submitquery', views.submitquery, name='submitquery'),
    path('scrape', views.scrape, name="scrape"),
    path('test', views.test, name="test"),
    path('upload', views.image_upload_view, name="image_upload_view"),
    path('home', views.home, name="home"),
    path('today', views.today, name="today"),
    path('analysis', views.analysis, name="analysis"),
    path('trial', views.trial, name="trial")
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

