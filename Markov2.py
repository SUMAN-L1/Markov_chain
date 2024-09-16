import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Streamlit file uploader
st.title("Markov Chain Analysis with Export Data")
uploaded_file = st.file_uploader("Upload your Excel file", type="xlsx")

# Error handling in case of issues with file upload or data processing
try:
    if uploaded_file is not None:
        # 1. Read the uploaded Excel file
        df = pd.read_excel(uploaded_file)

        # 2. Automatically remove the "Years" and "Total" columns (if they exist)
        if 'Years' in df.columns:
            df = df.drop(columns=['Years'])
        if 'Total' in df.columns:
            df = df.drop(columns=['Total'])

        # Display the uploaded data
        st.write("Uploaded Dataset:", df)

        # 3. Calculate the transition matrix
        # Normalize each row by the total for that year
        transition_matrix = df.div(df.sum(axis=1), axis=0)

        # Calculate transition probabilities from one year to the next (row-wise differences)
        tpm = transition_matrix.pct_change().fillna(0)

        # Ensure values are non-negative
        tpm[tpm < 0] = 0

        # Normalize to ensure that the rows sum up to 1
        tpm = tpm.div(tpm.sum(axis=1), axis=0).fillna(0)

        # Display the TPM
        st.write("Transition Probability Matrix (TPM):")
        st.dataframe(tpm)

        # 4. Plot the heatmap of the Transition Probability Matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(tpm.T, annot=True, cmap="YlGnBu", cbar=True)
        plt.title('Transition Probability Matrix Heatmap')
        plt.xlabel('Years')
        plt.ylabel('Countries')

        # Display the heatmap in Streamlit
        st.pyplot(plt)

except Exception as e:
    st.error(f"Error occurred: {e}")
