import pandas as pd
import numpy as np
import tensorflow as tf
from .teams import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import Perceptron, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB



"""
Creating a draw predictor model
Steps
filtering the data
overscaling and oversampling the data
Creating the model
Looping over a set of variables to find the best model
"""


class Data_Cleaning:
    ''' Predicts draw based on the last part'''
    
    def __init__(self, sample_weight=None):
        self.df = pd.read_csv('static/homeThis')
        self.train = self.df.iloc[: int(0.6*len(self.df))]
        self.val = self.df.iloc[int(0.6*len(self.df)):int(0.8*len(self.df))]
        self.test = self.df.iloc[int(0.8*len(self.df)):]
        self.train, self.x_train, self.y_train = self.scaling_data(self.train, oversample=True)
        self.val, self.x_val, self.y_val = self.scaling_data(self.val)
        self.test, self.x_test, self.y_test = self.scaling_data(self.test)
        self.newDf, self.x_newDf, self.y_newDf = self.scaling_data(self.df)
        self.sample_weight= sample_weight
        self.knn_model = KNeighborsClassifier()
        self.knn_model.fit(self.x_train, self.y_train)
        self.nb_model = GaussianNB()
        self.nb_model.fit(self.x_train, self.y_train, sample_weight=self.sample_weight)
        self.lr_model = LinearRegression()
        self.lr_model.fit(self.x_train, self.y_train, sample_weight=self.sample_weight)
        self.p_model = Perceptron()
        self.p_model.fit(self.x_train, self.y_train,sample_weight=self.sample_weight)
        self.svm_model = SVC()
        self.svm_model.fit(self.x_train, self.y_train, sample_weight=self.sample_weight)
        
    """def split(self):
        self.train = self.df.iloc[: int(0.6*len(self.df))]
        self.val = self.df.iloc[int(0.6*len(self.df)):int(0.8*len(self.df))]
        self.test = self.df.iloc[int(0.8*len(self.df)):]
    """
    def scaling_data(self, df, oversample=False):
        x = df[df.columns[:-1]].values
        y = df[df.columns[-1]].values
    
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
    
        if oversample:
            ros = RandomOverSampler()
            x, y = ros.fit_resample(x, y)
    
        data = np.hstack((x, np.reshape(y, (-1, 1))))
    
        return data, x, y
    
    #def scaled_data(self):
    #    self.train, self.x_train, self.y_train = self.scaling_data(self.train, oversample=True)
    #    self.val, self.x_val, self.y_val = self.scaling_data(self.val)
    #    self.test, self.x_test, self.y_test = self.scaling_data(self.test)
    
    """Naive Bayes model"""
    
    def nb(self):
        #val_predict = self.nb_model.predict(self.x_newDf)
        test_predict=self.nb_model.predict(self.x_newDf)
        return accuracy_score(self.y_newDf, test_predict)
    """Linear Regression"""
    
    def linear_reg(self):
       val_predict = self.lr_model.predict(self.x_test)
       test_predict=self.lr_model.predict(self.x_newDf)
       return classification_report(self.y_newDf, test_predict)
    """Perceptron"""
    def perceptron(self):
        val_predict= self.p_model.predict(self.x_val)
        test_predict= self.p_model.predict(self.x_newDf)
        return classification_report(self.y_newDf, test_predict)
    """SVM"""
    def svms(self):
        val_predict= self.svm_model.predict(self.x_val)
        test_predict=self.svm_model.predict(self.x_newDf)
        return accuracy_score(self.y_newDf, test_predict)
    def knn(self):
        val_predict = self.knn_model.predict(self.x_val)
        test_predict=self.knn_model.predict(self.x_newDf)
        return accuracy_score(self.y_newDf, test_predict)
    
    def tf_model(self):
    	model = tf.keras.models.load_model('static/fifaHomeWins.keras')
    	#val = model.evaluate(self.x_newDf, self.y_newDf)
    	loss, acc = model.evaluate(self.x_newDf, self.y_newDf)
    	return acc


class Output:
	def __init__(self, sample_weight=None):
		self.df = pd.read_csv('static/homeThis')
		self.train = self.df.iloc[: int(0.6*len(self.df))]
		self.val = self.df.iloc[int(0.6*len(self.df)):int(0.8*len(self.df))]
		self.test = self.df.iloc[int(0.8*len(self.df)):]
		self.train, self.x_train, self.y_train = Data_Cleaning().scaling_data(self.train, oversample=True)
		self.val, self.x_val, self.y_val = Data_Cleaning().scaling_data(self.val)
		self.test, self.x_test, self.y_test = Data_Cleaning().scaling_data(self.test)
		self.newDf, self.x_newDf, self.y_newDf = Data_Cleaning().scaling_data(self.df)
		self.sample_weight= sample_weight
		self.knn_model = KNeighborsClassifier()
		self.knn_model.fit(self.x_train, self.y_train)
		self.nb_model = GaussianNB()
		self.nb_model.fit(self.x_train, self.y_train, sample_weight=self.sample_weight)
		self.lr_model = LinearRegression()
		self.lr_model.fit(self.x_train, self.y_train, sample_weight=self.sample_weight)
		self.p_model = Perceptron()
		self.p_model.fit(self.x_train, self.y_train,sample_weight=self.sample_weight)
		self.svm_model = SVC()
		self.svm_model.fit(self.x_train, self.y_train, sample_weight=self.sample_weight)
		self.tf_model = tf.keras.models.load_model('static/fifaHomeWins.keras')
	
	"""Tensor flow model"""
	def tf_model_output(self):
		y_predict = self.tf_model.predict(self.x_newDf)
		return y_predict
	def svms_model_output(self):
		y_predict = self.svm_model.predict(self.x_newDf)
		return y_predict
	def knn_model_output(self):
		y_predict = self.svm_model.predict(self.x_newDf)
		return y_predict
	def nb_model(self):
		y_predict = self.svm_model.predict(self.x_newDf)
		return y_predict
	
		


class Reports:
    def nb_report(self):
        nb_report = Data_Cleaning(sample_weight=None).nb()
        print('\n nb_report')
        print(nb_report)
    def lr_report(self):
        lr_report = Data_Cleaning().linear_reg()
        print('lr_report')        
        print(lr_report)
    def Perceptron_report(self):
        p_report = Data_Cleaning().perceptron()
        print('perceptron')
        print(p_report)
    def svm_report(self):
        svc_report = Data_Cleaning().svms()
        print('svm_report')
        print(svc_report)
    def knn_report(self):
        knn_report = Data_Cleaning().knn()
        print('knn_report')
        print(knn_report)


"""Support Vector machines is the best model to 
use with an accuracy of 91% validation set
"""
def run_set():
    report = Reports()
    report.nb_report()
    report.Perceptron_report()
    report.svm_report()
    report.knn_report()


class SvmModel:
    """
    Take the output of the model(y_predict)
    locate the specific location of the model
    pick the home team of the prediction
    Pick the away Team of the prediction
    Display the home and away Team
    Ask The user for the home and away team
    loop over the home teams found and match it with away tema
    display the result if it exists
    """
    def __init__(self):
        self.old_df = pd.read_csv('static/homeThis')
        self.df, self.x_df, self.y_df = Data_Cleaning().scaling_data(self.old_df)
        self.model = Data_Cleaning().svm_model
        self.y_predict = self.model.predict(self.x_df)
        #self.homeTeam = self.UserInput().homeTeam
        #self.awayTeam = self.UserInput().awayTeam
        
    # We have the model
    def teams(self):
        """ 
        No_teams converts teams to numbers for the user to enter
        winDict converts numbers to Teams
        """
        for x in range(0, len(self.y_predict)):
            if self.old_df.iloc[x, -2] == 20:
                HomeTeamNo = self.old_df.iloc[x, 0]
                HomeTeamName = no_teams[HomeTeamNo]
                AwayTeamNo = self.old_df.iloc[x, 1]
                AwayTeamName = no_teams[AwayTeamNo]
                print('{} vs {} = draw'.format(HomeTeamName, AwayTeamName))
            else:
                if self.y_predict[x] == 1:
                    WinTeamNo = self.old_df.iloc[x, 0]
                    WinTeamName = no_teams[WinTeamNo]
                    HomeTeamName = WinTeamName
                    AwayTeamNo = self.old_df.iloc[x, 1]
                    AwayTeamName = no_teams[AwayTeamNo]
                    print("{} vs {} = {} ".format(HomeTeamName, AwayTeamName, 
                                                  WinTeamName))
                
                else:
                    
                    #if self.y_predict[x] == 0:
                    WinTeamNo = self.old_df.iloc[x, 1]
                    WinTeamName = no_teams[WinTeamNo]
                    AwayTeamName = WinTeamName
                    HomeTeamNo = self.old_df.iloc[x, 0]
                    HomeTeamName = no_teams[HomeTeamNo]
                    print('{} vs {} = {}'.format(HomeTeamName, AwayTeamName, 
                                                 WinTeamName))
                
        print(classification_report(self.y_df, self.y_predict))
        
    def UserInput(self):
#        homeTeam = input("\n\nHome Team: ")
#        awayTeam = input("AwayTeam: ")
        homeTeam = 'Chelsea'
        AwayTeam = 'Man City'
        #homeTeamNo = teamDict[homeTeam]
        #awayTeamNo = teamDict[awayTeam]
        """Loop over the values to identify Which team matches 
        with the other"""
        
        for x in range(0, len(self.y_df)):
            if self.old_df.iloc[x, -2] == 20:
                HomeTeamNo = self.old_df.iloc[x, 0]
                AwayTeamNo = self.old_df.iloc[x, 1]
                HomeTeamName = no_teams[HomeTeamNo]
                AwayTeamName = no_teams[AwayTeamNo]
                if HomeTeamName == homeTeam and AwayTeamName == awayTeam:
                    print('\n{} vs {} = draw'.format(homeTeam, awayTeam))
                
            else:
                if self.old_df.iloc[x, -1] == 1:
                    homeTeamNo = self.old_df.iloc[x, 0]
                    awayTeamNo = self.old_df.iloc[x, 1]
                    homeTeamName = no_teams[homeTeamNo]
                    winTeamName = homeTeamName
                    awayTeamName = no_teams[awayTeamNo]
                    if homeTeamName == homeTeam and awayTeamName == awayTeam:
                        print('\n{} vs {} = {}'.format(homeTeam,
                                                     awayTeam, winTeamName))
                else:
                    homeTeamNo = self.old_df.iloc[x, 0]
                    awayTeamNo = self.old_df.iloc[x, 1]
                    homeTeamName = no_teams[homeTeamNo]
                    awayTeamName = no_teams[awayTeamNo]
                    winTeamName = awayTeamName
                    if HomeTeamName == homeTeam and awayTeamName == awayTeam:
                        print('\n{} vs {} = {}'.format(homeTeam, awayTeam, 
                                                     winTeamName))
                        """
        #print('Actual Result')
        for x in range(len(self.y_predict)):
            if self.y_df[x] == 1:
                  winTeam_no = self.old_df.iloc[x, 0]
                  winTeam_name = no_teams[winTeam_no]
                  homeTeam_no = self.old_df.iloc[x, 0]
                  homeTeam_name = no_teams[homeTeam_no]
                  awayTeam_no = self.old_df.iloc[x, 1]
                  awayTeam_name = no_teams[awayTeam_no]
            if homeTeam_name == homeTeam and awayTeam_name == awayTeam:
                print('\n{} vs {} = {}'.format(homeTeam_name, awayTeam_name, winTeam_name))
                      
            else:
                if df.iloc[x, -1] == 1:
                    homeTeam_no = self.old_df.iloc[x, 0]
                    homeTeam_name = no_teams[homeTeam_no]
                    awayTeam_no = self.old_df.iloc[x, 1]
                    awayTeam_name = no_teams[awayTeam_no]
                    if homeTeam_name == HomeTeam and awayTeam_name == AwayTeam:
                        print('{} vs {} = Draw'.format(homeTeam_name, awayTeam_name))
                        
                else:
                    winTeam_no = self.old_df.iloc[x, 1]
                    winTeam_name = no_teams[winTeam_no]
                    homeTeam_no = self.old_df.iloc[x, 0]
                    homeTeam_name = no_teams[homeTeam_no]
                    awayTeam_no = self.old_df.iloc[x, 1]
                    awayTeam_name = no_teams[awayTeam_no]
                    if homeTeam_name == homeTeam and awayTeam_name == awayTeam:
                        print('{} vs {} = {}'.format(homeTeam_name, awayTeam_name, winTeam_name))
            """
    def looping_userInput(self):
        while True:
            #self.homeTeam
            #self.awayTeam
            self.UserInput()

if __name__ == '__main__':
    report = Reports()
#    report.svm_report()
    #run_set()
    svm = Svm_Model()
    #svm.teams()
    svm.looping_userInput()

