from django import forms
from django.forms import TextInput

class TeamSelection(forms.Form):
	homeTeam = forms.CharField(required=True, max_length=20,
		    label = 'Home Team',
		    widget=TextInput(attrs={
		    	'placeholder': 'Enter The Home Team',
		    	'style': 'font-size: large; margin-top: 40px; margin-bottom: 40px; margin-left: 10px;   	margin-right: 20px;'
		    }))
	awayTeam = forms.CharField(required=True, max_length=20,
		label='Away team',
	widget=TextInput(attrs={
		    	'placeholder': 'Enter The Home Team',
		    	'style': 'font-size: large; margin-top: 40px; margin-bottom: 40px; margin-left: 10px;   	margin-right: 20px;'
		    }))
	
