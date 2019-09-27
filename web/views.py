import json

from django.http import HttpResponse, Http404
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from prediction_Module.mnist_nn_predict import predict
from .forms import ImageUploadForm


# Create your views here.


def index(request):
    if request.method == 'POST':
        print(request.POST)
        print(request.FILES)
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_object = form.save(commit=True)

            name = image_object.image.name.split('/')[1]
            path = image_object.image.url

            label = predict(image_object.image.path)
            return render(request, 'index.html', locals())
        else:
            raise Http404
    return render(request, 'index.html')

