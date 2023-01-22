import numpy as np
import os
import PIL
import tempfile
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import time

from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from keras.models import load_model
from pathlib import Path

from .forms import ImageForm, ImageFormPrune
from .models import Image


def index(request):
    return render(request, "base.html")


def pruning(model, initial_sparsity, target_sparsity, begin_step, end_step, granularity, pruning_type, scheduling,
            mode):
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    # Compute end step to finish pruning after 2 epochs.
    batch_size = 128
    epochs = 2
    validation_split = 0.1  # 10% of training set will be used for validation set.

    # Count number of images used for training
    dirs_count = 0
    files_count = 0
    directory = Path(__file__).resolve().parent.parent
    for path, dirs, files in os.walk(os.path.join(directory, "databases/Big_Suspect_Database/train")):
        dirs_count += len(dirs)
        files_count += len(files)

    num_images = files_count * (1 - validation_split)
    # end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

    # Define model for pruning.
    if pruning_type == "polynomial":
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=initial_sparsity,
                                                                     final_sparsity=target_sparsity,
                                                                     begin_step=begin_step,
                                                                     end_step=end_step)
        }
    else:
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=target_sparsity,
                                                                      begin_step=begin_step,
                                                                      end_step=end_step)
        }

    model_for_pruning = prune_low_magnitude(model, **pruning_params)

    # 'prune_low_magnitude' requires a recompile.
    model_for_pruning.compile(optimizer='adam',
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                              metrics=['accuracy'])
    model_for_pruning.summary()

    log_dir = tempfile.mkdtemp()
    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir=log_dir),
    ]

    batch_size = 128  # @param [1,2,4,8,16,32,64,128] {type:"raw"}
    epochs = 2  # @param [2, 5, 10,20,50,100,200] {type:"raw"}
    dataset_name = os.path.join(directory, 'databases/Big_Suspect_Database')
    train_dataset = os.path.join(dataset_name, 'train/')
    test_dataset = os.path.join(dataset_name, 'test/')
    input_dim = 255  # @param [224,299] {type:"raw"}

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dataset,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(input_dim, input_dim),
        batch_size=batch_size,
        label_mode='categorical',
    )

    train_ds = train_ds.unbatch()
    train_images = list(train_ds.map(lambda x, y: x))
    train_labels = list(train_ds.map(lambda x, y: y))

    print("*" * 10 + " TRAIN")
    model_for_pruning.fit(train_images, train_labels,
                          batch_size=batch_size, epochs=epochs,
                          validation_split=validation_split,
                          callbacks=callbacks)

    return model_for_pruning


@login_required
def image_upload_view(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            img_obj = form.instance
            # ============== machine learning
            model = load_model("Model_Suspect_Detection.h5")
            # model = pruning(load_model("Model_Suspect_Detection.h5"))

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
            model_size = os.path.getsize("Model_Suspect_Detection.h5")

            return render(request, 'upload.html', {'form': form,
                                                   'class_name': class_final,
                                                   'probability': probability,
                                                   'elapsed': elapsed,
                                                   'model_size': model_size,
                                                   })
    else:
        form = ImageForm()
    return render(request, 'upload.html', {'form': form})


@login_required
def image_upload_prune_view(request):
    if request.method == 'POST':
        form = ImageFormPrune(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            img_obj = form.instance

            # ============== machine learning
            # model = load_model("Model_Suspect_Detection.h5")
            model = pruning(load_model("Model_Suspect_Detection.h5"), form.instance.initial_sparsity,
                            form.instance.target_sparsity, form.instance.begin_step, form.instance.end_step,
                            form.instance.granularity, form.instance.pruning_type, form.instance.scheduling,
                            form.instance.mode)

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
            model_size = os.path.getsize("Pruned.h5")

            return render(request, 'upload.html', {'form': form,
                                                   'class_name': class_final,
                                                   'probability': probability,
                                                   'elapsed': elapsed,
                                                   'model_size': model_size,
                                                   })
    else:
        form = ImageFormPrune()
    return render(request, 'upload.html', {'form': form})
