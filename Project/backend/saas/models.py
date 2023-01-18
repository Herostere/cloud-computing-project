import os
import uuid

from django.db import models


IRR = "IRREGULAR"
REG = "REGULAR"

LOC = "LOCAL"
GLO = "GLOBAL"

ONE = "ONE SHOT"
CON = "CONSTANT"
POL = "POLYNOMIAL"

MAG = "MAGNITUDE"
MOV = "MOVEMENT"

GRANULARITY_CHOICES = (
    (IRR, "Irregular"),
    (REG, "Regular"),
)

PRUNING_TYPE_CHOICES = (
    (LOC, "Local"),
    (GLO, "Global"),
)

SCHEDULING_CHOICES = (
    (ONE, "One shot"),
    (CON, "Constant"),
    (POL, "Polynomial"),
)

MODE_CHOICES = (
    (MAG, "Magnitude"),
    (MOV, "Movement"),
)


def update_filename(instance, filename):
    path = "images/"
    extension = filename.split(".")[-1]
    format = str(uuid.uuid4()) + f".{extension}"
    return os.path.join(path, format)


class Image(models.Model):
    image = models.ImageField(upload_to=update_filename, blank=True)
    initial_sparsity = models.FloatField()
    target_sparsity = models.FloatField()
    begin_step = models.IntegerField(default=0)
    end_step = models.IntegerField(default=0)
    granularity = models.CharField(max_length=9, choices=GRANULARITY_CHOICES, default="IRREGULAR")
    pruning_type = models.CharField(max_length=6, choices=PRUNING_TYPE_CHOICES, default="LOCAL")
    scheduling = models.CharField(max_length=10, choices=SCHEDULING_CHOICES, default="ONE SHOT")
    mode = models.CharField(max_length=9, choices=MODE_CHOICES, default="MAGNITUDE")
