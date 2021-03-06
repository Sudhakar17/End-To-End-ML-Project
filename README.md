# End-To-End-ML-Project

In-order to understand the machine learning work flow, I followed the tutorials from the 2nd chapter of Hands-on-ML using Scikit-learn and Tensorflow by Aurelien Geron. 

## Dataset

We will use the California Housing Prices [dataset](https://www.kaggle.com/camnugent/california-housing-prices) from the StatLib repository. This dataset is based on data from the 1990 California census.

## Steps to follow for the Machine learning application.

1. Look at the big picture.
2. Get the data.
3. Discover and visualize the data to gain insights.
4. Prepare the data for Machine Learning algorithms.
5. Select a model and train it.
6. Fine-tune your model.
7. Present your solution.
8. Launch, monitor, and maintain your system.

The first seven steps are performed in this [Jupyter Notebook](https://github.com/Sudhakar17/End-To-End-ML-Project/blob/master/End-To-End-ML-Project.ipynb)

## Deployment of the ML model using Flask 

  * [app.py](https://github.com/Sudhakar17/End-To-End-ML-Project/blob/master/app.py) contains the Flask API which gets the features as an input through an API call and request from HTML page.
  * [utils/price_calculator.py](https://github.com/Sudhakar17/End-To-End-ML-Project/blob/master/utils/price_calculator.py) is the python file which does the preprocessing of the inputs from the HTML page and calculate the median housing price of the house.
  * [templates/main.html](https://github.com/Sudhakar17/End-To-End-ML-Project/blob/master/templates/main.html) allows user to enter the inputs and display the results.

## Requirements:
  * Flask
  * NumPy
  * Pandas
  * Scikit-Learn
  
## How to run this flask application

  * After cloning this repository, run app.py using the following command to start the flask API
    * python app.py
  * Navigate to URL http://localhost:5000
    * By default, flask will run on port 5000.
  * You can see the HTML page as below
    ![Input Variables](images/input-variables.png)
  * You can use the [sample_test_input.json](sample_test_input.json) for entering the input variables.
  * After submitting, the calculated median house price is shown like below.
    ![Results](images/results.png)







