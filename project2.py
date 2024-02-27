import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Read the CSV file into a pandas DataFrame
df = pd.read_csv('C:\\Users\\DAN\\Documents\\ITCS 3162-Data Mining\\Projects\\Project 2\\emails.csv')




#################################################################
#Pie chart function

def plot_prediction_distribution(dataframe):
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
plot_prediction_distribution(df)


######################################################################
#Confusion Matrix

#Filtering the amount of rows and columns 
#df_subset = df.iloc[-50:, -50:]

# Target variable is in the last column
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

# Training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

#Confusion Matrix creation
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()