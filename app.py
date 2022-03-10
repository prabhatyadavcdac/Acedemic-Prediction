import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect, flash, send_file
from sklearn.preprocessing import MinMaxScaler
from werkzeug.utils import secure_filename
import pickle
#from xgboost import XGBClassifier
#import xgboost 
#import xgboost
#import xgboost.compat
#from xgboost.compat import XGBoostLabelEncoder
#import xgboost as xgb

import numpy as np
from flask import Flask, request, jsonify
import pickle
import os



app = Flask(__name__) #Initialize the flask App
app = Flask(__name__)
#model = pickle.load(open('model.pkl', 'rb'))
rf = pickle.load(open('rf.pkl', 'rb'))
@app.route('/')


@app.route('/index')
def index():
	return render_template('login.html')

import webbrowser

@app.route('/test')
def my_page():
    return webbrowser.open_new_tab('http://mylink.com')

@app.route('/personal_behavior')
def personal_behavior():
	return render_template('personal_behavior.html')

@app.route('/stu_sta')
def stu_sta():
	return render_template('stu_sta.html')

@app.route('/lifestyle')
def lifestyle():
	return render_template('lifestyle.html')

#@app.route('/future')
#def future():
#	return render_template('future.html')    

@app.errorhandler(500)
def internal_error(error):

    return "<h1>500 error</h1>"

@app.errorhandler(404)
def not_found(error):
    return "<h1>404 error</h1>",404

@app.route('/login')
def login():
	return render_template('login.html')



@app.route('/marks')
def marks():
	return render_template('prediction.html')


@app.route('/learning')
def learning():
    return render_template('learning.html')  
	
@app.route('/preview', methods=['GET', 'POST'])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset,encoding = 'unicode_escape')
        df.set_index('Id', inplace=True)
        return render_template("preview.html",df_view = df)	


#@app.route('/home')
#def home():
 #   return render_template('home.html')

@app.route('/prediction', methods = ['GET', 'POST'])
def prediction():
    return render_template('marks.html')
@app.route('/final', methods = ['GET', 'POST'])
def final():

	# output = format(prediction[0])
	if (output == 'good'):
		return render_template('final.html', prediction_text= 'Can Opt For Engineer ')
	else :
		return render_template('final.html', prediction_text= 'Can Opt For Doctor ')




@app.route('/train')
def train():
    return render_template('train.html')  

#@app.route('/upload')
#def upload_file():
#   return render_template('BatchPredict.html')

@app.route('/predict',methods=['POST'])
def predict():
	int_feature = [x for x in request.form.values()]
	print(int_feature)
	int_feature = [float(i) for i in int_feature]
	final_features = [np.array(int_feature)]
	prediction = rf.predict(final_features)
 
	output = format(prediction[0])
	print(output)
	print(output)
	print(output)
	print(output)
	
	if output == "0":
		return render_template('prediction.html', prediction_text= 'Can GO For Engineer ')
	elif output == "1" :
		return render_template('prediction.html', prediction_text= 'Can GO For Milatary ')

	else  :
		return render_template('prediction.html', prediction_text= 'Can GO For Doctor ')



# @app.route('/predict',methods=['POST'])

# def predict():
# data = request.get_json(force=True)
# predict_request=[[data['sepal_length'],data['sepal_width'],data['petal_length'],data['petal_width']]]
# request=np.array(predict_request)
# print(request)
# prediction = load_model.predict(predict_request)
# pred = prediction[0]
# print(pred)
# return jsonify(int(pred))
# 	if (output == 'good'):
# 		return render_template('prediction.html', prediction_text= 'Can +Opt For Engineer or Doctor')
# 	else :
# 		return render_template('prediction.html', prediction_text= 'Higher Efforts Required ')


# @app.route('/predict6',methods=['POST'])
# def predict():
# 	int_feature = [x for x in request.form.values()]
# 	print(int_feature)
# 	int_feature = [float(i) for i in int_feature]
# 	final_features = [np.array(int_feature)]
# 	prediction = rf.predict(final_features)
 
# 	output = format(prediction[0])
# 	print(output)
# 	print(output)
# 	print(output)
# 	print(output)
	
# 	if output == "0":
# 		return render_template('prediction.html', prediction_text= 'Can GO For Engineer ')
# 	elif output == "1" :
# 		return render_template('prediction.html', prediction_text= 'Can GO For Milatary ')

# 	else  :
# 		return render_template('prediction.html', prediction_text= 'Can GO For Doctor ')



@app.route('/chart')
def chart():
	return render_template('chart.html')  
@app.route('/future')
def future():
	return render_template('future.html')     
    
if __name__ == "__main__":
    app.run(debug=True)
	
