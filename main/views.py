from django.shortcuts import render, redirect
from django.http import JsonResponse

# Create your views here.
def main(request):
    return render(request, 'main.html')