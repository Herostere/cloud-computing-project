from django.urls import path

from .views import image_upload_view, image_upload_prune_view, index


urlpatterns = [
    path('', index, name='index'),
    path('upload/', image_upload_view, name='upload'),
    path('upload_prune/', image_upload_prune_view, name='upload_prune'),
]
