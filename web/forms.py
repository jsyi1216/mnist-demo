from django.forms import ModelForm
from django import forms
from .models import ImageUpload

class ImageUploadForm(ModelForm):

    class Meta:
        model = ImageUpload
        fields = '__all__'