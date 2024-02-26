import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Create and train a machine learning model
model = RandomForestClassifier()
model.fit(X, y)

# Streamlit app code
def predict_iris_species(sepal_length, sepal_width, petal_length, petal_width):
    data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(data)
    species = iris.target_names[prediction][0]
    return species

def main():
    st.title('Iris Dataset Prediction')
    st.write('Please enter the following features to get the predicted species.')

    sepal_length = st.slider('Sepal Length', min_value=0.0, max_value=10.0, step=0.1)
    sepal_width = st.slider('Sepal Width', min_value=0.0, max_value=10.0, step=0.1)
    petal_length = st.slider('Petal Length', min_value=0.0, max_value=10.0, step=0.1)
    petal_width = st.slider('Petal Width', min_value=0.0, max_value=10.0, step=0.1)

    if st.button('Predict'):
        prediction = predict_iris_species(sepal_length, sepal_width, petal_length, petal_width)
        st.write(f'Predicted Iris species: {prediction}')

if __name__ == '__main__':
    main()