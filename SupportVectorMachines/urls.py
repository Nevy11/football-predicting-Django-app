from django.urls import path
from . import views

urlpatterns = [
	path("", views.home, name='home'),
	path('results/<str:homeTeam>/<str:awayTeam>/', views.results, name='results'),
	path('tf_output/<str:homeTeam>/<str:awayTeam>/', views.tf_output, name='tf_output'),
#	path('test/', views.testing, name='testing'),
]
