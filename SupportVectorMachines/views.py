import pandas as pd
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.template import loader
from math import ceil
from .files import svm
from .files.svm import Output
from .forms import *



def home(request):

	form = TeamSelection()
	template = loader.get_template('home.html')
	with open('static/Teams.txt', 'r') as f:
		teams = f.read()
	naive_bayes = svm.Data_Cleaning().nb()
	#Linear_reg = svm.Data_Cleaning().linear_reg()
	svms = svm.Data_Cleaning().svms()
	knn = svm.Data_Cleaning().knn()
	#teams = svm.SvmModel().UserInput()
	acc = svm.Data_Cleaning().tf_model()
	context = {
		'teams': teams,
		'svms': svms * 100,
		'knn': knn * 100,
		'nb': naive_bayes * 100,
		'form': form,
		'tf': acc * 100,
	#	'UserInput': teams,
	}
	if request.method == 'POST':
		form = TeamSelection(request.POST)
		if form.is_valid():
			homeTeam = form.cleaned_data['homeTeam']
			awayTeam = form.cleaned_data['awayTeam']
			return redirect('tf_output', homeTeam = homeTeam, awayTeam = awayTeam)
	else:
		form = TeamSelection(request.POST)
		return HttpResponse(template.render(context, request))


def results(request, homeTeam, awayTeam):
	template = loader.get_template('results.html')
	y_pred_tf = Output().tf_model_output()
	df = pd.read_csv('static/homeThis')
	cols = df.columns[-1]
	
	"""Choosing the home and away team"""
	
	
	
	context = {
		'y_pred_tf': y_pred_tf,
		'cols': cols
	}
	return HttpResponse(template.render(context, request))
	


def tf_output(request, homeTeam, awayTeam):
	template = loader.get_template('tf_output.html')
	y_pred = Output().tf_model_output()
	df = pd.read_csv('static/homeThis')
	teamDf = pd.read_csv('static/eng.csv')
	cols = df.columns[-1]
	"""Teams"""
	teams = teamDf[df.columns[1]].unique()
	teamToIndex = {}
	for i, team in enumerate(teams):
		teamToIndex[team] = i

	#print(teamToIndex)
	indexToTeam = {}
	for i, team in enumerate(teams):
		indexToTeam[i] = team
	#print(indexToTeam)
	for x in range(0, len(y_pred)):
		if ceil(y_pred[x]) == 1:
			"""Home Team wins"""
			homeTeamNo = df.iloc[x, 0]
			awayTeamNo = df.iloc[x, 1]
			homeTeamName = indexToTeam[homeTeamNo]
			awayTeamName = indexToTeam[awayTeamNo]
			winTeam = homeTeamName
			
			if homeTeamName == homeTeam and awayTeamName == awayTeam:
				context = { 
						'y_pred': y_pred,
						'cols': cols,
						'homeTeam': homeTeam,
						'awayTeam': awayTeam,
						'winTeam' : winTeam,
					}
				return HttpResponse(template.render(context, request))
				break
		else:
			if df.iloc[x, -2] == 20:
				homeTeamNo = df.iloc[x, 0]
				awayTeamNo = df.iloc[x, 1]
				homeTeamName = indexToTeam[homeTeamNo]
				awayTeamName = indexToTeam[awayTeamNo]
				if homeTeamName == homeTeam and awayTeamName == awayTeam:
					context = { 
						'y_pred': y_pred,
						'cols': cols,
						'homeTeam': homeTeam,
						'awayTeam': awayTeam,
						'winTeam' : winTeam,
					}
					return HttpResponse(template.render(context, request))
				
			else:
				homeTeamNo = df.iloc[x, 0]
				awayTeamNo = df.iloc[x, 1]
				homeTeamName = indexToTeam[homeTeamNo]
				awayTeamName = indexToTeam[awayTeamNo]
				winTeam = awayTeamName
				if homeTeamName == homeTeam and awayTeamName == awayTeam:
					context = { 
						'y_pred': y_pred,
						'cols': cols,
						'homeTeam': homeTeam,
						'awayTeam': awayTeam,
						'winTeam' : winTeam,
					}
					return HttpResponse(template.render(context, request))
	context = {
		'y_pred': y_pred,
		'cols': cols
	}
	"""Y result"""
	
	return HttpResponse(template.render(context, request))
	
