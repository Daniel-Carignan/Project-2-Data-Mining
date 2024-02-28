import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

################################################################
# All this is needed for the model to work
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Gets the CSV frile we are looking at
df = pd.read_csv('C:\\Users\\DAN\\Documents\\ITCS 3162-Data Mining\\Projects\\Project 2\\emails.csv')


#################################################################
#Pie chart function creates pie chart from the data

def pie_chart(dataframe):
    #Prediction column
    prediction_column = 'Prediction'

    # Count the occurrences of each value in the 'Prediction' column
    prediction_counts = dataframe[prediction_column].value_counts()

    # Keeps the right values to the right names
    prediction_labels = {0: 'Not Spam', 1: 'Spam'}
    prediction_counts.index = prediction_counts.index.map(prediction_labels)

    # Pie chart configuration
    plt.figure(figsize=(8, 8))
    plt.pie(prediction_counts, labels=prediction_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Prediction Distribution')
    plt.show()

# Call the function with your DataFrame
pie_chart(df)


######################################################################


# Target the column we are trying to figure out
target_column = 'Prediction'

# This selects the columns with numeric values making it easier to deal with
numeric_features = df.select_dtypes(include=['number'])

# Excludes the target column from features
features = numeric_features.drop(columns=[target_column])

# Initialize the Gaussian Naive Bayes model
model = GaussianNB()

# Training 'features' based off of 'target_column'
model.fit(features, df[target_column])

# Make predictions on the entire dataset from the trained model 'features'
predictions = model.predict(features)

# Evaluates the model
accuracy = accuracy_score(df[target_column], predictions)
conf_matrix = confusion_matrix(df[target_column], predictions)
classification_rep = classification_report(df[target_column], predictions)


# Displays the Confusion Matrix for visual
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
print(f'Accuracy: {accuracy:.2f}')
print('\nClassification Report:')
print(classification_rep)
