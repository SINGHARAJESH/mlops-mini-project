from flask import Flask, render_template,request
import mlflow
from preprocessing_utility import normalize_text
import dagshub
import pickle
mlflow.set_tracking_uri("https://dagshub.com/rajeshai2000/mlops-mini-project.mlflow")
dagshub.init(repo_owner='rajeshai2000', repo_name='mlops-mini-project', mlflow=True)

app = Flask(__name__)
model_name = "my_model"
model_version = 3
model_url = f"models:/{model_name}/{model_version}"
model = mlflow.pyfunc.load_model(model_url)

vectorizer = pickle.load(open('models/vectorizer.pkl','rb'))
@app.route('/')

def home():
    return render_template('index.html',result = None)

@app.route('/predict',methods = ['POST'])

def predict():
    text = request.form['text']
    text = normalize_text(text)
    features = vectorizer.transform([text])
    result = model.predict(features)
    return render_template('index.html',result=result[0])
    

app.run()