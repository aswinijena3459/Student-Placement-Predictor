from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
import sklearn

app=Flask(__name__)

@app.route('/')
def Home():
    return render_template('index.html')

@app.route('/predict',methods=(['POST','GET']))
def predict():
    cgpa=float(request.form['cgpa'])
    iq=int(request.form['iq'])
    profile_score=int(request.form['profile_score'])

    x=np.array([[cgpa,iq,profile_score]])
    model = pickle.load(open('model.pkl', 'rb'))
    result=model.predict(x)[0]

    return jsonify({"placement":str(result)})

if __name__=='__main__':
    app.run(debug=True,port=8000)
