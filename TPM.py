import streamlit as st
import pandas as pd
import numpy as np
from functools import reduce
from math import gcd
from itertools import combinations

# Define the Markov Chain class with methods
class MarkovChain:
    def __init__(self, transition_matrix, states):
        self.transition_matrix = np.atleast_2d(transition_matrix)
        self.states = states
        self.index_dict = {self.states[index]: index for index in range(len(self.states))}
        self.state_dict = {index: self.states[index] for index in range(len(self.states))}

    def next_state(self, current_state):
        return np.random.choice(self.states, p=self.transition_matrix[self.index_dict[current_state], :])

    def generate_states(self, current_state, no=10):
        future_states = []
        for i in range(no):
            next_state = self.next_state(current_state)
            future_states.append(next_state)
            current_state = next_state
        return future_states

    def is_accessible(self, i_state, f_state, check_up_to_depth=1000):
        counter = 0
        reachable_states = [self.index_dict[i_state]]
        for state in reachable_states:
            if counter == check_up_to_depth:
                break
            if state == self.index_dict[f_state]:
                return True
            else:
                reachable_states.extend(np.nonzero(self.transition_matrix[state, :])[0])
            counter = counter + 1
        return False

    def is_irreducible(self):
        for (i, j) in combinations(self.states, 2):
            if not self.is_accessible(i, j):
                return False
        return True

    def get_period(self, state, max_number_stps=50, max_number_trls=100):
        initial_state = state
        max_number_steps = max_number_stps
        max_number_trials = max_number_trls
        periodic_lengths = []

        for i in range(1, max_number_steps + 1):
            for j in range(max_number_trials):
                last_states_chain = self.generate_states(current_state=initial_state, no=i)[-1]
                if last_states_chain == initial_state:
                    periodic_lengths.append(i)
                    break

        if len(periodic_lengths) > 0:
            a = reduce(gcd, periodic_lengths)
            return a

    def is_aperiodic(self):
        periods = [self.get_period(state) for state in self.states]
        for period in periods:
            if period != 1:
                return False
        return True

    def is_transient(self, state):
        if np.all(self.transition_matrix[~self.index_dict[state], self.index_dict[state]] == 0):
            return True
        else:
            return False

    def is_absorbing(self, state):
        state_index = self.index_dict[state]
        if self.transition_matrix[state_index, state_index] == 1:
            return True
        else:
            return False

# Streamlit app setup
st.title("Markov Chain Analysis for Silk Production Data")

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV file with silk production data", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data preview:", df.head())

    # Assuming the columns are countries and rows are time series data
    if not df.empty:
        # Create a Markov Chain model
        st.write("Calculating transition matrix...")
        
        # Create a transition matrix
        data = df.apply(lambda x: x.astype('category').cat.codes)
        states = df.columns
        transition_matrix = np.zeros((len(states), len(states)))
        
        for i in range(len(states)):
            for j in range(len(states)):
                transition_matrix[i, j] = np.mean(data[states[i]].shift(-1) == data[states[j]])
        
        markov_chain = MarkovChain(transition_matrix, states)

        st.write("Transition Matrix:")
        st.write(pd.DataFrame(transition_matrix, index=states, columns=states))

        # State selection and analysis
        selected_state = st.selectbox("Select a state (country)", states)

        if st.button("Generate Future States"):
            no_of_states = st.slider("Number of future states to generate", 1, 100, 10)
            future_states = markov_chain.generate_states(selected_state, no=no_of_states)
            st.write("Generated Future States:", future_states)

        if st.button("Check State Properties"):
            st.write(f"Is '{selected_state}' accessible from other states?", markov_chain.is_accessible(states[0], selected_state))
            st.write(f"Is '{selected_state}' irreducible?", markov_chain.is_irreducible())
            st.write(f"Period of '{selected_state}':", markov_chain.get_period(selected_state))
            st.write(f"Is '{selected_state}' aperiodic?", markov_chain.is_aperiodic())
            st.write(f"Is '{selected_state}' transient?", markov_chain.is_transient(selected_state))
            st.write(f"Is '{selected_state}' absorbing?", markov_chain.is_absorbing(selected_state))
