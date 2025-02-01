Marriage Eligibility Prediction

Overview

This project predicts marriage eligibility based on the provided dataset. It utilizes a Decision Tree Classifier to analyze the relationship between age of marriage and gender and determine eligibility.

Features

Reads a dataset containing age_of_marriage and gender.

Converts gender to numerical labels using LabelEncoder.

Determines eligibility based on median age of marriage for each gender.

Uses a Decision Tree Classifier to train the model.

Evaluates the model's accuracy using a test dataset.

Provides a visualization of age of marriage distribution per gender.

Accepts user input for age and gender to predict eligibility.

Technologies Used

Google Colab

Python

Pandas

NumPy

Matplotlib & Seaborn (for visualization)

Scikit-learn (for machine learning)

Usage in Google Colab

Open Google Colab: Google Colab

Upload your dataset (marriage.csv) to the Colab environment.

Run the following command to install required dependencies:

!pip install pandas numpy matplotlib seaborn scikit-learn

Load your dataset into a Pandas DataFrame:

from google.colab import files
uploaded = files.upload()
import pandas as pd
dataset = pd.read_csv('marriage.csv')

Execute the script in Colab by running the provided Python code cells.

Enter the requested details (age of marriage & gender) when prompted.

The model will predict and display whether you are eligible for marriage.

Visualization

The project includes a histogram displaying the distribution of age of marriage by gender, along with median age indicators for better analysis.

Accuracy

The trained model evaluates its accuracy using a test dataset split. The accuracy score is displayed in the output cell of Colab.

Contributing

Feel free to submit issues or pull requests to enhance the project!

License

This project is licensed under the MIT License.

AUTHOR

ELURU POOJITH KUMAR REDDY


