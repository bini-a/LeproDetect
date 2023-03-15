import streamlit as st
from joblib import load
import pandas as pd
def app():
    st.write("# Operational Classification")
    with st.sidebar:
        # Get user's age as a slider input
        age = st.slider("Enter your age", 0, 100, 25)
        gender = st.selectbox("Choose Gender", ["Male", "Female"])
        num_lesion = st.number_input("Enter number of Lesions", step = 1)
        num_affected_nerves = st.number_input("Enter number of Affected Nerves", step = 1)
        bacillo_exam = st.selectbox("Choose Bacilloscopy exam", [1,2,3])
        grade_disability = st.selectbox("Choose Grade of disability", [0,1,2,3])

    # Define a dictionary to store the input values
    input_dict = {}
    # Get user input and store in dictionary
    input_dict['Patient age'] = age
    input_dict['Gender'] = gender
    input_dict["Number of lesions"] = num_lesion
    input_dict[ 'Grade of disability'] = 1 if grade_disability==0 else 0
    input_dict["bacilloscopy exam"] = bacillo_exam
    input_dict["Number of Affected Nerves"] = num_affected_nerves

# 1 if gender == "Female" else 2
    st.write("### Your Input:")
    df_input = pd.DataFrame(input_dict.items(), columns=["Feature", "Value"])
    df_input.Feature = df_input["Feature"].str.capitalize()
    st.write(df_input.style.set_table_styles([
        {'selector': 'th', 'props': [('max-width', '100px')]},
        {'selector': 'td', 'props': [('max-width', '100px')]}
    ]))    # st.table(df_input.style.set_caption("Input Features"))

    @st.cache_resource
    def load_model():
        return load("model/model-operational.joblib")

    if st.button("Generate prediction"):
        model = load_model()
        inp = input_dict.copy()
        inp["Gender"]= 1 if gender == "Female" else 2
        X_test = pd.DataFrame(inp, index=[0])
        y_pred = model.predict(X_test)[0]
        prediction_text = "Paucibacillary" if y_pred ==1 else "Multibacillary"
        st.write("#### <em>Operational Type:   </em>" + prediction_text, unsafe_allow_html=True)
