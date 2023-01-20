from django import forms
from .models import Image, ImagePrune


class ImageFormPrune(forms.ModelForm):
    initial_sparsity = forms.FloatField(min_value=0.0, max_value=1.0, initial=0.0)
    target_sparsity = forms.FloatField(min_value=0.0, max_value=1.0, initial=0.0)
    begin_step = ImagePrune.begin_step
    end_step = ImagePrune.end_step
    granularity = ImagePrune.granularity
    pruning_type = ImagePrune.pruning_type
    scheduling = ImagePrune.scheduling
    mode = ImagePrune.mode

    class Meta:
        model = ImagePrune
        fields = ("image", "initial_sparsity", "target_sparsity", "begin_step", "end_step", "granularity",
                  "pruning_type", "scheduling", "mode")


class ImageForm(forms.ModelForm):
    class Meta:
        model = Image
        fields = ("image",)
