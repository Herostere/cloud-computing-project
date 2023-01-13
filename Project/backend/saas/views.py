from django.http import HttpResponse


def index(request):
    return HttpResponse("Hello, friend. It is the main page.")
