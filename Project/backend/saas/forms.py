from django import forms
from .models import Image


class ImageForm(forms.ModelForm):
    initial_sparsity = forms.FloatField(min_value=0.0, max_value=1.0, initial=0.0)
    target_sparsity = forms.FloatField(min_value=0.0, max_value=1.0, initial=0.0)
    begin_step = Image.begin_step
    end_step = Image.end_step
    granularity = Image.granularity
    pruning_type = Image.pruning_type
    scheduling = Image.scheduling
    mode = Image.mode

    class Meta:
        model = Image
        fields = ("image", "initial_sparsity", "target_sparsity", "begin_step", "end_step", "granularity",
                  "pruning_type", "scheduling", "mode")
