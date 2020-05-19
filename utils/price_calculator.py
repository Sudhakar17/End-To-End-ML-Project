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

def preprocessing_pipeline(input_dict):
    
    all_features   = np.array([float(v) for k, v in input_dict.items()])
    print(all_features)
    num_features = all_features[:-1]
    attr_adder = CombinedAttributesAdder()
    total_num_features = attr_adder.transform(num_features)
    
    # Mean of the features from the training samples
    mean_train_set = np.array([-1.19575834e+02,  3.56395773e+01,  2.86531008e+01,  2.62272832e+03,
        5.33998123e+02,  1.41979082e+03,  4.97060380e+02,  3.87558937e+00,
        5.44034053e+00,  3.09643738e+00,  2.13703119e-01])

    # Variance of the features from the training samples 
    var_train_set = np.array([4.00720174e+00, 4.57101322e+00, 1.58114157e+02, 4.57272746e+06,
       1.68778972e+05, 1.24468040e+06, 1.41157604e+05, 3.62861320e+00,
       6.82062550e+00, 1.34200064e+02, 4.26971514e-03])
   
    scaled_num_features = (total_num_features - mean_train_set) / np.sqrt(var_train_set)
    

    ocean_proximity_feature = {'1H OCEAN':0, 'INLAND':1, 'NEAR OCEAN':2,
                                'NEAR BAY':3, 'ISLAND':4}
    total_ocean_categories = len(ocean_proximity_feature)
    cat_feature  = input_dict['ocean_proximity']
    one_hot_ocean_feature = np.eye(total_ocean_categories)[int(cat_feature)]


    total_features = np.hstack((scaled_num_features,one_hot_ocean_feature)).reshape(1,-1)

    return total_features




def predict_results(input_dict):
    with open('.\\models\\final_model_grid.pkl', 'rb') as f:
        model = joblib.load(f)
    total_features = preprocessing_pipeline(input_dict)
    predictions = model.predict(total_features)
    return predictions[0]