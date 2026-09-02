from django.urls import path
from . import views


urlpatterns = [

    # Home Page
    path(
        "",
        views.home,
        name="home"
    ),

    # Prediction
    path(
        "predict/",
        views.predict,
        name="predict"
    ),

    # Prediction History
    path(
        "history/",
        views.history,
        name="history"
    ),

    # Chatbot
    path(
        "chatbot/",
        views.chatbot,
        name="chatbot"
    ),

]