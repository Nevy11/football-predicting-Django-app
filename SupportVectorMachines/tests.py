from django.test import TestCase
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import tensorflow as tf
import pandas as pd
import numpy as np

model = tf.keras.models.load_model('fifaHomeWins.keras')
df = pd.read_csv('/media/nevy11/samma_rs/projects/web_dev/django/FootballPrediction/static/homeThis')
def scale_dataset(df, oversample=False):
	x = df[df.columns[:-1]].values
	y = df[df.columns[-1]].values
	
	scaler = StandardScaler()
	x = scaler.fit_transform(x)
	
	if oversample:
		ros = RandomOverSampler()
		x, y = ros.fit_resample(x, y)
	
	data = np.hstack((x, np.reshape(y, (-1, 1))))
	return data, x, y

df, x_df, y_df = scale_dataset(df)

loss, acc = model.evaluate(x_df, y_df)
print('Accuracy: {} %'.format(acc*100))

df = pd.read_csv('/media/nevy11/samma_rs/projects/web_dev/django/FootballPrediction/static/eng.csv')
print(df.head())
cols = df.columns[-1]
"""Teams"""
teams = df[df.columns[1]].unique()
teamToIndex = {}
for i, team in enumerate(teams):
	teamToIndex[team] = i

print(teamToIndex)
indexToTeam = {}
for i, team in enumerate(teams):
	indexToTeam[i] = team
print(indexToTeam)
