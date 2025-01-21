import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import streamlit as st

# Streamlit App Title
st.title("Customer Churn Prediction App")

# Load Dataset
st.header("Step 1: Load Dataset")
url = "https://raw.githubusercontent.com/abilpandoli/Bank-Customer-Churn/main/E%20Commerce%20Dataset.xlsx"
table_name = "E Comm"
data = pd.read_excel(url, engine='openpyxl', sheet_name=table_name)
st.write("Loaded Dataset:", data.head())

# Data Cleaning
st.header("Step 2: Data Cleaning")
object_columns = data.select_dtypes(include=['object'])
label_encoders = {}
for col in object_columns.columns:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col].astype(str))
st.write("Data After Encoding Object Columns:", data.head())

# Handle Missing Values
imputer = SimpleImputer(strategy='median')
data.iloc[:, :] = imputer.fit_transform(data)
st.write("Data After Imputing Missing Values:", data.head())

# Exclude 'CustomerID'
data = data.drop(columns=['CustomerID'])
st.write("Data After Excluding 'CustomerID':", data.head())

# Split Data into Train and Test
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)
train_target = train_data.pop('Churn')
test_target = test_data.pop('Churn')
st.write(f"Training Data Shape: {train_data.shape}")
st.write(f"Test Data Shape: {test_data.shape}")

# Normalize Features
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(tf.convert_to_tensor(train_data))

# Define Model
def get_basic_model():
    model = tf.keras.Sequential([
        normalizer,
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

# Train Model
st.header("Step 3: Train Model")
model = get_basic_model()
model.fit(tf.convert_to_tensor(train_data), train_target, epochs=10, batch_size=128, verbose=1)
st.success("Model Training Complete!")

# Evaluate Model on Test Data
test_normalizer = tf.keras.layers.Normalization(axis=-1)
test_normalizer.adapt(tf.convert_to_tensor(test_data))
score = model.evaluate(tf.convert_to_tensor(test_data), test_target, verbose=1)
st.write(f"Test Loss: {score[0]:.4f}")
st.write(f"Test Accuracy: {score[1]:.4f}")

# Predict on New Data
st.header("Step 4: Make Predictions")
predict_data = data.sample(10, random_state=42)
predict_target = predict_data.pop('Churn')
numeric_predict_features = tf.convert_to_tensor(predict_data)
predictions = model(numeric_predict_features, training=False)

# Add Predictions to DataFrame
predict_data['Predicted_Label'] = tf.squeeze(tf.cast(tf.nn.sigmoid(predictions) > 0.5, tf.int32)).numpy()
predict_data['Actual_Label'] = predict_target.values
st.write("Predictions on Sample Data:", predict_data)

st.header("Thank You for Using the App!")
