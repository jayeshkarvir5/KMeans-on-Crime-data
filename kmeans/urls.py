from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('state/', views.selected_option, name="state"),
    path('state/cluster_size', views.clustering, name="kval"),

]
