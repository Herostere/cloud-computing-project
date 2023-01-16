import numpy as np
import os
import PIL
import time
import tensorflow as tf

from django.shortcuts import render
from keras import backend
from keras.models import load_model

from .forms import ImageForm
from .models import Image


def index(request):
    return render(request, "base.html")


def image_upload_view(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            img_obj = form.instance
            # ============== machine learning
            model = load_model("Model_Suspect_Detection.h5")
            input_dim = 224
            classes = [
                'appareil photo', 'arme', 'autre', 'baton', 'couteau', 'drone', 'gilet jaune', 'grenade', 'personne'
            ]
            classes_translated = {
                "appareil photo": "camera",
                "arme": "weapon",
                "autre": "other",
                "baton": "stick",
                "couteau": "knife",
                "drone": "drone",
                "gilet jaune": "yellow vest",
                "grenade": "grenade",
                "personne": "people",
            }

            image_to_test = "media/" + str(img_obj.image)

            img = PIL.Image.open(image_to_test).convert('RGB')
            x = tf.keras.utils.img_to_array(img, data_format='channels_last')
            x = tf.keras.preprocessing.image.smart_resize(x, size=(input_dim, input_dim))
            x = np.expand_dims(x, axis=0)
            start = time.perf_counter()
            prediction = model.predict(x, batch_size=64)[0]
            end = time.perf_counter()

            elapsed = round(end - start, 2)

            class_final = ""
            probability = 0

            for (pos, prob) in enumerate(prediction):
                class_name = classes[pos]
                if pos == np.argmax(prediction):
                    class_final = class_name
                    probability = round(prob * 100, 2)
            # ==============
            class_final = classes_translated[class_final]

            os.remove(image_to_test)
            Image.objects.all().delete()
            return render(request, 'upload.html', {'form': form,
                                                   'class_name': class_final,
                                                   'probability': probability,
                                                   'elapsed': elapsed,
                                                   })
    else:
        form = ImageForm()
    return render(request, 'upload.html', {'form': form})
