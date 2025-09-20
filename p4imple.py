# app.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor


st.title("Simple Conversion (Yes/No) + Revenue Prediction App")


@st.cache_resource
def load_and_train():
    
    df = pd.read_csv("p4train.csv")

    
    features = ["year", "month", "day", "order", "session_id", "page",
                "country", "page1_main_category", "page2_clothing_model",
                "colour", "location", "model_photography"]

    X = df[features]
    y_class = (df["price_2"] == 2).astype(int)   
    y_reg = df["price"]                         

    
    num_cols = ["year", "month", "day", "order", "session_id", "page"]
    cat_cols = [c for c in features if c not in num_cols]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    
    clf = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestClassifier(class_weight="balanced", random_state=42))
    ])
    clf.fit(X, y_class)

    
    reg = Pipeline([
        ("prep", preprocessor),
        ("model", GradientBoostingRegressor(random_state=42))
    ])
    reg.fit(X, y_reg)

    return clf, reg, features

clf, reg, features = load_and_train()


st.header("Enter details manually")

manual = {}
for col in features:
    if col in ["year", "month", "day", "order", "session_id", "page"]:
        manual[col] = st.number_input(f"{col}", value=1)
    else:
        manual[col] = st.text_input(f"{col}", value="Sample")

if st.button("Predict"):
    
    input_df = pd.DataFrame([manual])

    
    prob = clf.predict_proba(input_df)[0][1]
    pred = "YES" if prob >= 0.5 else "NO"
    revenue = reg.predict(input_df)[0]

    st.write("### Prediction Results")
    st.write(f"Predicted Conversion: **{pred}**")
    st.write(f"Conversion Probability: {prob*100:.2f}%")
    st.write(f"Predicted Revenue: {revenue:.2f}")
