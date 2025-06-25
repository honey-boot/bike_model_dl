# 🚴‍♂️ Bike Rental Prediction (Deep Learning Regression with Streamlit)

This project predicts the number of bike rentals based on weather and time features using a deep learning regression model built with TensorFlow and deployed via Streamlit.

---

## 📌 Project Overview

- **Goal**: Predict hourly bike rental counts using historical data.
- **Dataset**: UCI Bike Sharing Dataset (`hour.csv`)
- **Model**: Deep Learning Regression model using Keras (TensorFlow backend)
- **Deployment**: Streamlit Web App

---

## 🧠 Features Used

The model uses the following 13 features:

- `season`, `yr`, `mnth`, `hr`, `holiday`, `weekday`, `workingday`, `weathersit`
- `temp`, `atemp`, `hum`, `windspeed`, `hour`

---

## 🚀 How to Run the App

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/bike-rental-prediction.git
cd bike-rental-prediction

2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Run Streamlit App
bash
Copy
Edit
streamlit run app.py
🧾 Requirements
nginx
Copy
Edit
streamlit
tensorflow
scikit-learn
pandas
numpy
(Also available in requirements.txt)

🗂️ Project Structure
graphql
Copy
Edit
📁 bike-rental-prediction/
├── app.py               # Streamlit app
├── bike_model.h5        # Trained deep learning model
├── scaler.pkl           # Saved StandardScaler for input preprocessing
├── hour.csv             # Raw dataset (optional)
├── requirements.txt     # Python dependencies
└── README.md            # Project overview
📈 Model Info
Architecture: 3 Dense layers with ReLU activations and Dropout

Loss Function: MSE (Mean Squared Error)

Optimizer: Adam

Evaluation Metric: MAE (Mean Absolute Error)

📊 Sample UI Screenshot
<!-- Optional: add a screenshot -->


