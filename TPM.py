import streamlit as st
import numpy as np
import pandas as pd

def calculate_transition_matrix(df):
    # Calculate the number of states (columns in the dataframe)
    n_states = df.shape[1]
    
    # Initialize the transition matrix with zeros
    transition_matrix = np.zeros((n_states, n_states))
    
    # Loop over the rows and columns to populate the transition matrix
    for i in range(len(df) - 1):
        for j in range(n_states):
            current_state = df.iloc[i, j]
            next_state = df.iloc[i + 1, j]
            transition_matrix[j, :] += np.histogram(next_state, bins=n_states, range=(0, n_states))[0]
    
    # Normalize the transition matrix
    transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
    
    return transition_matrix

def main():
    st.title("Markov Chain Analysis of Crop Price Data")
    
    # Upload file
    uploaded_file = st.file_uploader("Upload your file (CSV or Excel)", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        # Read the uploaded file into a DataFrame
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Show the uploaded data
        st.subheader("Uploaded Data")
        st.write(df)
        
        # Calculate the transition probability matrix
        transition_matrix = calculate_transition_matrix(df)
        
        # Convert to a pandas DataFrame for better readability
        transition_matrix_df = pd.DataFrame(transition_matrix, index=df.columns, columns=df.columns)
        
        # Show the transition probability matrix
        st.subheader("Transition Probability Matrix")
        st.write(transition_matrix_df)

if __name__ == "__main__":
    main()
