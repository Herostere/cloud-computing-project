from django import forms
from .models import Image, ImagePrune


class ImageFormPrune(forms.ModelForm):
    image = ImagePrune.image
    initial_sparsity = forms.FloatField(min_value=0.0, max_value=1.0, initial=0.0)
    target_sparsity = forms.FloatField(min_value=0.0, max_value=1.0, initial=0.0)
    begin_step = ImagePrune.begin_step
    end_step = ImagePrune.end_step
    granularity = ImagePrune.granularity
    pruning_type = ImagePrune.pruning_type
    scheduling = ImagePrune.scheduling
    mode = ImagePrune.mode

    def __init__(self, *args, **kwargs):
        super(ImageFormPrune, self).__init__(*args, **kwargs)
        self.fields["image"].widget.attrs.update({"class": "btn btn-default btn-block"})
        self.fields["image"].label = False
        self.fields["initial_sparsity"].widget.attrs.update({"class": "form-control"})
        self.fields["target_sparsity"].widget.attrs.update({"class": "form-control"})
        self.fields["begin_step"].widget.attrs.update({"class": "form-control"})
        self.fields["end_step"].widget.attrs.update({"class": "form-control"})
        self.fields["granularity"].widget.attrs.update({"class": "form-control"})
        self.fields["pruning_type"].widget.attrs.update({"class": "form-control"})
        self.fields["scheduling"].widget.attrs.update({"class": "form-control"})
        self.fields["mode"].widget.attrs.update({"class": "form-control"})

    class Meta:
        model = ImagePrune
        fields = ("image", "initial_sparsity", "target_sparsity", "begin_step", "end_step", "granularity",
                  "pruning_type", "scheduling", "mode")


class ImageForm(forms.ModelForm):
    image = Image.image

    def __init__(self, *args, **kwargs):
        super(ImageForm, self).__init__(*args, **kwargs)
        self.fields["image"].widget.attrs.update({"class": "btn btn-default btn-block"})
        self.fields["image"].label = False

    class Meta:
        model = Image
        fields = ("image",)
