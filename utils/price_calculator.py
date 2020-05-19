import pandas as pd 
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.room_ix = 3
        self.bedrooms_ix = 4
        self.population_ix = 5
        self.household_ix = 6

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        rooms_per_household = x[self.room_ix] / x[self.household_ix]
        population_per_household = x[self.population_ix] / x[self.household_ix]
        bedrooms_per_room = x[self.bedrooms_ix]/ x[self.room_ix]
        new_features = np.array((rooms_per_household, population_per_household, bedrooms_per_room))
        x = np.hstack((x, new_features))
        return x





def predict_results(input_dict):
    with open('.\\models\\final_model_grid.pkl', 'rb') as f:
        model = joblib.load(f)
    total_features = preprocessing_pipeline(input_dict)
    predictions = model.predict(total_features)
    return predictions[0]