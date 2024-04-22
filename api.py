import flask
from flask import Flask, render_template, request,make_response
import joblib
#from sklearn.externals import joblib
import inputScript
import regex
import mysql.connector
from mysql.connector import Error
import random
import sys
import logging
from sklearn.metrics import confusionmatrix
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import predictor
import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import json


app = Flask(__name__)

app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index')
def indexnew():    
    return render_template('index.html')

@app.route('/register')
def register():    
    return render_template('register.html')


@app.route('/login')
def login():
    return render_template('login.html')




""" REGISTER CODE  """

@app.route('/regdata', methods =  ['GET','POST'])
def regdata():
    connection = mysql.connector.connect(host='localhost',database='flaskphishingdb',user='root',password='')
    uname = request.args['uname']
    name = request.args['name']
    pswd = request.args['pswd']
    email = request.args['email']
    phone = request.args['phone']
    addr = request.args['addr']
    value = random.randint(123, 99999)
    uid="User"+str(value)
    print(addr)
        
    cursor = connection.cursor()
    sql_Query = "insert into userdata values('"+uid+"','"+uname+"','"+name+"','"+pswd+"','"+email+"','"+phone+"','"+addr+"')"
        
    cursor.execute(sql_Query)
    connection.commit() 
    connection.close()
    cursor.close()
    msg="Data stored successfully"
    #msg = json.dumps(msg)
    resp = make_response(json.dumps(msg))
    
    print(msg, flush=True)
    #return render_template('register.html',data=msg)
    return resp




"""LOGIN CODE """

@app.route('/logdata', methods =  ['GET','POST'])
def logdata():
    import datetime
    current_time = datetime.datetime.now()
    if current_time.day>31:
        msg="Failure"
        resp = make_response(json.dumps(msg))
        return resp
    connection=mysql.connector.connect(host='localhost',database='flaskphishingdb',user='root',password='')
    lgemail=request.args['email']
    lgpssword=request.args['pswd']
    print(lgemail, flush=True)
    print(lgpssword, flush=True)
    cursor = connection.cursor()
    sq_query="select count(*) from userdata where Email='"+lgemail+"' and Pswd='"+lgpssword+"'"
    cursor.execute(sq_query)
    data = cursor.fetchall()
    print("Query : "+str(sq_query), flush=True)
    rcount = int(data[0][0])
    print(rcount, flush=True)
    
    connection.commit() 
    connection.close()
    cursor.close()
    
    if rcount>0:
        msg="Success"
        resp = make_response(json.dumps(msg))
        return resp
    else:
        msg="Failure"
        resp = make_response(json.dumps(msg))
        return resp
        
   


'''

@app.route('/')
def dataloaders():
    return render_template('dataloader.html')
'''

@app.route('/dataloader')
def dataloader():
    return render_template('home.html')

@app.route('/about')
def about():
    return flask.render_template('about.html')

@app.route('/predict', methods = ['POST'])
def make_prediction():
    classifier = joblib.load('rf_final.pkl')
    if request.method=='POST':
        url = request.form['url']
        if not url:
            return render_template('home.html', label = 'Please input url')
        elif(not(regex.search(r'^(http|ftp)s?://', url))):
            return render_template('home.html', label = 'Please input full url, for exp- https://facebook.com')
        
        
        checkprediction = inputScript.main(url)
        prediction = classifier.predict(checkprediction)

        if prediction[0]==1 :
            label = 'website is not legitimate'
        elif prediction[0]==-1:
            label ='website is legitimate'

        lracc=predictor.logistic_regression()
        nbacc=predictor.navie_bayes()
        hyacc=predictor.hybrid()


        arr_random = np.random.randint(low=0, high=2, size=(1000,4))
        testdataframe=pd.DataFrame(arr_random, columns=["A","B","C","target"])
        print(type(testdataframe))
        ddataset=testdataframe
        a="NB"
        try:
            nb = GaussianNB()

            nb.fit(X_train,Y_train)

            Y_pred_nb = nb.predict(X_test)
        except:
            nb = GaussianNB()
            
            
        nbscore=confusionmatrix.accuracy_score()
        labels=[0,1]
        X, Y = make_classification(n_samples=500, n_classes=2, n_features=20, random_state=10)
        data=confusionmatrix.heatmap(X,labels)
        actual,predicted=confusionmatrix.generatelabels(ddataset,X,labels,a)
        len(actual)
        # confusion matrix
        matrix = confusion_matrix(actual,predicted)
        print('Confusion matrix : \n',matrix)

        matrix = classification_report(actual,predicted)
        print('Classification report : \n',matrix)



        from sklearn import metrics
        confusion_matrix1 = metrics.confusion_matrix(actual, predicted)
        '''
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix1, display_labels = labels)
        cm_display.plot()
        plt.savefig("Nbcon.png")
        plt.show()
        '''

        a="LR"
        try:
            nb = LogisticRegression()

            nb.fit(X_train,Y_train)

            Y_pred_nb = nb.predict(X_test)
        except:
            nb = LogisticRegression()
            
            
        nbscore=confusionmatrix.accuracy_score()
        #Confusion Matrix
        labels=[0,1]
        X, Y = make_classification(n_samples=500, n_classes=2, n_features=20, random_state=10)
        data=confusionmatrix.heatmap(X,labels)
        actual,predicted=confusionmatrix.generatelabels(ddataset,X,labels,a)
        len(actual)
        # confusion matrix
        matrix = confusion_matrix(actual,predicted)
        print('Confusion matrix : \n',matrix)

        matrix = classification_report(actual,predicted)
        print('Classification report : \n',matrix)

        a="LRRF"
        try:
            nb = LogisticRegression()

            nb.fit(X_train,Y_train)

            Y_pred_nb = nb.predict(X_test)
        except:
            nb = LogisticRegression()
            
            
        nbscore=confusionmatrix.accuracy_score()
        #Confusion Matrix
        labels=[0,1]
        X, Y = make_classification(n_samples=500, n_classes=2, n_features=20, random_state=10)
        data=confusionmatrix.heatmap(X,labels)
        actual,predicted=confusionmatrix.generatelabels(ddataset,X,labels,a)
        len(actual)
        # confusion matrix
        matrix = confusion_matrix(actual,predicted)
        print('Confusion matrix : \n',matrix)

        matrix = classification_report(actual,predicted)
        print('Classification report : \n',matrix)

        from sklearn import metrics
        confusion_matrix1 = metrics.confusion_matrix(actual, predicted)
        '''
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix1, display_labels = labels)
        cm_display.plot()
        plt.savefig("lrcon.png")
        plt.show()
        '''

                
        
        
        return render_template('home.html', label=label)
        
        
if __name__ == '__main__':
    classifier = joblib.load('rf_final.pkl')
    app.run()
