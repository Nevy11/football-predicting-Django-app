import pandas as pd

df = pd.read_csv('eng.csv')

teams = df.columns[1]


teams = df[teams].unique()


teams = list(teams)
print('The following is a list of teams entered')
for x in range(len(teams)):
	print(teams[x])

index = list((i for i in range(0, len(teams)-1)))


# Teams Dictionary
teamDict = {}
for i in range(0, len(teams)-1):
    teamDict[teams[i]] = index[i]
    #print(teams, teamDict[teams[i]])

no_teams = {}
for i in range(0, len(teams)-1):
	no_teams[index[i]] = teams[i]
	#print(i, no_teams[index[i]])
no_teams[19] = 'Luton'


for i in range(0, len(teams)-1):
	k = enumerate(teamDict)
	#print(list(k))
"""
homeTeam = input("Enter Home team: ")
awayTeam = input('Enter away team: ')
"""

homeTeam = 'Chelsea'
awayTeam = 'Man City'	



if homeTeam not in teamDict:
	print("The home team entered is not in the dictionary")


if homeTeam not in teamDict:
	print("The away team entered is not in the dictionary")
	

homeTeam = teamDict.get(homeTeam)
awayTeam = teamDict.get(awayTeam)



#print(homeTeam)
#print(awayTeam)
"""
# Referees 


refs = list(df["Referee"].unique())
index = list(i for i in range(0, len(refs)))

refDict = {}

for i in index:
    refDict[refs[i]] = i



ref = input("Enter Referee name: ")

#ref = 'A Taylor'

if ref not in refDict:
	print("The home team entered is not in the dictionary")



ref = refDict[ref]
"""

# winner
# converting winner back to Strings
teams = df.columns[-1]
#print(teams)
teams = list(df[teams].unique())
#print(teams)
index = list( i for i in range(len(teams)+1))
winDict = {}
winDict["Bournemouth"] = 21
for i in range(0, len(teams)):
    winDict[index[i]] = teams[i]
    #print(i , winDict[index[i]]) 


   
winner = 2
winner = winDict

