import os
import uuid

from django.db import models


def update_filename(instance, filename):
    path = "images/"
    extension = filename.split(".")[-1]
    format = str(uuid.uuid4()) + f".{extension}"
    return os.path.join(path, format)


class Image(models.Model):
    image = models.ImageField(upload_to=update_filename, blank=True)
