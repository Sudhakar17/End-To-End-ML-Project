import flask
import pandas as pd
import joblib
import json
import utils.price_calculator as pc


app = flask.Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    return flask.render_template('main.html')


@app.route('/', methods=['POST'])
def main():

    if flask.request.method == 'POST':
        input_dict = flask.request.form.to_dict()
        print("Flask Inputs:{}".format(input_dict))
        predictions = pc.predict_results(input_dict)
        print(predictions)
        
        return flask.render_template('main.html',
                                     result=round(predictions))


if __name__ == '__main__':
    app.run(debug=True)
