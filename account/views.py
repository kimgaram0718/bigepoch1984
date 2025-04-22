from django.shortcuts import render, redirect
from django.http import JsonResponse

# Create your views here.
def signup(request):
    return render(request, 'signup.html')

def login(request):
    return render(request, 'login.html')