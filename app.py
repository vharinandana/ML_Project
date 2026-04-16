import streamlit as st
import pickle
import pandas as pd
import joblib
from PIL import Image

def main():

    st.set_page_config(page_title="Bank Churn Prediction")

    st.title("🏦 Bank Customer Churn Prediction")
    st.write("Predict whether a bank customer will leave the bank or not.")

    image = Image.open("bank.jpg")
    st.image(image, use_container_width=True)

    st.write("Enter customer details to predict churn")

    # Load model and encoders
    model = pickle.load(open("churn_model.pkl","rb"))
    country_enc = pickle.load(open("country_encoder.pkl","rb"))
    gender_enc = pickle.load(open("gender_encoder.pkl","rb"))

    # Load model feature columns
    model_columns = joblib.load("model_columns.pkl")

    # User Inputs
    credit_score = st.number_input("Credit Score", 300, 900)

    country = st.selectbox("Country", ["France","Germany","Spain"])

    gender = st.selectbox("Gender", ["Male","Female"])

    age = st.number_input("Age", 18, 100)

    tenure = st.selectbox("Tenure (Years)", [0,1,2,3,4,5,6,7,8,9,10])

    balance = st.number_input("Balance")

    products = st.selectbox("Number of Products", [1,2,3,4])

    credit_card = st.selectbox("Has Credit Card", ["Yes","No"])

    active_member = st.selectbox("Active Member", ["Yes","No"])

    salary = st.number_input("Estimated Salary")

    # Convert Yes/No to numeric
    credit_card = 1 if credit_card == "Yes" else 0
    active_member = 1 if active_member == "Yes" else 0

    if st.button("Predict Churn"):

        # One hot encoding
        country_encoded = country_enc.transform([[country]])
        gender_encoded = gender_enc.transform([[gender]])

        country_df = pd.DataFrame(
            country_encoded,
            columns=country_enc.get_feature_names_out()
        )

        gender_df = pd.DataFrame(
            gender_encoded,
            columns=gender_enc.get_feature_names_out()
        )

        # Numeric features
        num_df = pd.DataFrame([[credit_score, age, tenure, balance,products, credit_card, active_member, salary]],
                              columns=["credit_score","age","tenure","balance","products_number","credit_card",
                                "active_member","estimated_salary"])

        # Combine all features
        final_data = pd.concat([num_df, country_df, gender_df], axis=1)

        # Match model training columns
        final_data = final_data.reindex(columns=model_columns, fill_value=0)

        prediction = model.predict(final_data)

        if prediction[0] == 1:
            st.error("⚠️ Customer is likely to CHURN")
        else:
            st.success("✅ Customer is likely to STAY")

if __name__ == "__main__":
    main()