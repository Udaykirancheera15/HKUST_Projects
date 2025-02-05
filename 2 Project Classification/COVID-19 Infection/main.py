import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Dense

# Load datasets
def load_data():
    # Define column names based on the dataset description
    column_names = [
        'Fever', 'Tiredness', 'Dry-Cough', 'Difficulty-in-Breathing', 'Sore-Throat',
        'Pains', 'Nasal-Congestion', 'Runny-Nose', 'Diarrhea', 'Age', 'Gender', 'Country', 'Infected'
    ]
    
    # Load first dataset with 'infected' label
    first_data = pd.read_csv('firstData.txt', delimiter='\t', header=None, names=column_names)
    # Load second dataset without 'infected' label
    second_data = pd.read_csv('secondData.txt', delimiter='\t', header=None, names=column_names[:-1])
    return first_data, second_data

# Preprocess data
def preprocess_data(data):
    # Print column names and first few rows for debugging
    print("Columns in the dataset:", data.columns.tolist())
    print("First few rows of the dataset:\n", data.head())

    # Check if required columns exist
    required_columns = ['Gender', 'Country', 'Age']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' is missing in the dataset.")

    # Encoding categorical variables
    # Gender
    le_gender = LabelEncoder()
    data['Gender'] = le_gender.fit_transform(data['Gender'])
    # Country
    le_country = LabelEncoder()
    data['Country'] = le_country.fit_transform(data['Country'])
    # Age
    le_age = LabelEncoder()
    data['Age'] = le_age.fit_transform(data['Age'])
    return data

# Prepare datasets
try:
    first_data, second_data = load_data()
    first_data = preprocess_data(first_data)
    second_data = preprocess_data(second_data)
except Exception as e:
    print("Error during data loading or preprocessing:", e)
    exit()

# Separate features and target
if 'Infected' not in first_data.columns:
    raise ValueError("Column 'Infected' is missing in the first dataset.")
X = first_data.drop('Infected', axis=1)
y = first_data['Infected']

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# Model 1: Logistic Regression
model1 = LogisticRegression()
model1.fit(X_train, y_train)
pred1 = model1.predict(second_data)
np.savetxt('predicted1.txt', pred1, fmt='%d')

# Model 2: Decision Tree
model2 = DecisionTreeClassifier()
model2.fit(X_train, y_train)
pred2 = model2.predict(second_data)
np.savetxt('predicted2.txt', pred2, fmt='%d')

# Model 3: Random Forest
model3 = RandomForestClassifier()
model3.fit(X_train, y_train)
pred3 = model3.predict(second_data)
np.savetxt('predicted3.txt', pred3, fmt='%d')

# Model 4: SVM
model4 = SVC(probability=True)
model4.fit(X_train, y_train)
pred4 = model4.predict(second_data)
np.savetxt('predicted4.txt', pred4, fmt='%d')

# Model 5: Neural Network
model5 = Sequential()
model5.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
model5.add(Dense(16, activation='relu'))
model5.add(Dense(1, activation='sigmoid'))
model5.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model5.fit(X_train, y_train, epochs=10, batch_size=32)
pred5 = (model5.predict(second_data) > 0.5).astype(int)
np.savetxt('predicted5.txt', pred5, fmt='%d')
