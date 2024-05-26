from sklearn.metrics import explained_variance_score,r2_score
#from sklearn.metrics import mean_squared_error
import pandas as pd

class Backtesting:
	def __init__(self, original_ytrain, train_predict, original_ytest, test_predict):
		
		self.original_ytrain = original_ytrain
		self.train_predict = train_predict

		self.original_ytest = original_ytest
		self.test_predict = test_predict

	def run(self):	
		'''
		Explained variance regression score:
		The explained variance score explains the dispersion of errors of a given dataset, 
		and the formula is written as follows: 
		Here, and Var(y) is the variance of prediction errors and actual values respectively. 
		Scores close to 1.0 are highly desired, indicating better squares of standard deviations of errors.
		
		print('Variance score:')
		print("Train data explained variance regression score:", explained_variance_score(self.original_ytrain, self.train_predict))
		print("Test data explained variance regression score:", explained_variance_score(self.original_ytest, self.test_predict))
		print('=================')


		R2 score for regression
		R-squared (R2) is a statistical measure that represents the proportion of the variance 
		for a dependent variable that's explained by an independent variable or variables in 
		a regression model
		1 = Best
		0 or < 0 = worse
		
		print('R2 score:')
		print("Train data R2 score:", r2_score(self.original_ytrain, self.train_predict))
		print("Test data R2 score:", r2_score(self.original_ytest, self.test_predict))
		print('=================')

	
		
		print('Mean squared error:')
		print("Train data accuracy regression score:", mean_squared_error(self.original_ytrain, self.train_predict))
		print("Test data accuracy regression score:", mean_squared_error(self.original_ytest, self.test_predict))
		print('=================')
		'''
		res = pd.DataFrame(index=['metrics'])
		res['train_variance_score'] = explained_variance_score(self.original_ytrain, self.train_predict)
		res['test_variance_score'] = explained_variance_score(self.original_ytest, self.test_predict)

		res['train_r2_score'] = r2_score(self.original_ytrain, self.train_predict)
		res['test_r2_score'] = r2_score(self.original_ytest, self.test_predict)


		res=res.reset_index(drop=True)
		
		return res