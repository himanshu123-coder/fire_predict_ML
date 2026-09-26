import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from  sklearn.preprocessing import StandardScaler


## import ridge regressio and standard Scaler pickle

ridge_model = pickle.load(open('models/ridge.pkl','rb'))
standard_scaler= pickle.load(open('models/scaler.pkl','rb'))



application = Flask(__name__)
app = application



@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():

    if request.method == "POST":

        Temperature = float(request.form.get("Temperature"))
        RH = float(request.form.get("RH"))
        Ws = float(request.form.get("Ws"))
        Rain = float(request.form.get("Rain"))
        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        ISI = float(request.form.get("ISI"))
        Classes = float(request.form.get("Classes"))
        Region = float(request.form.get("Region"))

        input_data = [[
            Temperature,
            RH,
            Ws,
            Rain,
            FFMC,
            DMC,
            ISI,
            Classes,
            Region
        ]]

        new_data_scaled = standard_scaler.transform(input_data)

        result = ridge_model.predict(new_data_scaled)

        return render_template(
            "predict.html",
            results=result[0]
        )

    return render_template("predict.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0")

  