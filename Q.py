import streamlit as st

# App Title
st.title("Multiplication App")

# Input Fields
num1 = st.number_input("Enter first number", value=0.0)
num2 = st.number_input("Enter second number", value=0.0)

# Multiplication Result
if st.button("Multiply"):
    result = num1 * num2
    st.success(f"Result: {result}")
