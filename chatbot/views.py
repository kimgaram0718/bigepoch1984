from django.shortcuts import render

# Create your views here.
def chatbot_view(request):
    return render(request, 'chatbot.html')