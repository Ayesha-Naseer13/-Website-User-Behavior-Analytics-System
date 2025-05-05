import numpy as np
import pandas as pd
from typing import List, Tuple

class MarkovChain:
    """
    Markov Chain model for analyzing website user navigation behavior.
    """
    
    def __init__(self, states: List[str], transition_matrix: np.ndarray = None):
        """
        Initialize the Markov Chain model.
        
        Args:
            states: List of state names (website pages)
            transition_matrix: Initial transition probability matrix (optional)
        """
        self.states = states
        self.n_states = len(states)
        
        if transition_matrix is not None:
            self.transition_matrix = transition_matrix
        else:
            # Initialize with uniform transition probabilities
            self.transition_matrix = np.ones((self.n_states, self.n_states)) / self.n_states
    
    def fit(self, sessions: List[List[str]]) -> None:
        """
        Estimate transition probabilities from observed user sessions.
        
        Args:
            sessions: List of user navigation sessions, where each session is a list of page names
        """
        # Initialize count matrix
        count_matrix = np.zeros((self.n_states, self.n_states))
        
        # Count transitions
        for session in sessions:
            for i in range(len(session) - 1):
                from_idx = self.states.index(session[i])
                to_idx = self.states.index(session[i + 1])
                count_matrix[from_idx, to_idx] += 1
        
        # Convert counts to probabilities
        row_sums = count_matrix.sum(axis=1, keepdims=True)
        
        # Handle zero rows (no outgoing transitions)
        zero_rows = (row_sums == 0).flatten()
        if np.any(zero_rows):
            # For states with no observed transitions, use uniform distribution
            count_matrix[zero_rows, :] = 1.0
            row_sums[zero_rows] = self.n_states
        
        self.transition_matrix = count_matrix / row_sums
    
    def generate_session(self, length: int, start_state: str = None) -> List[str]:
        """
        Generate a random user session based on the transition probabilities.
        
        Args:
            length: Number of pages in the session
            start_state: Initial state (page) for the session
            
        Returns:
            List of page names representing the generated session
        """
        if start_state is None:
            # Start from a random state
            start_idx = np.random.randint(0, self.n_states)
        else:
            start_idx = self.states.index(start_state)
        
        session = [self.states[start_idx]]
        current_idx = start_idx
        
        for _ in range(length - 1):
            # Sample next state based on transition probabilities
            next_idx = np.random.choice(self.n_states, p=self.transition_matrix[current_idx])
            session.append(self.states[next_idx])
            current_idx = next_idx
        
        return session
    
    def calculate_steady_state(self) -> np.ndarray:
        """
        Calculate the steady-state probabilities of the Markov chain.
        
        Returns:
            Array of steady-state probabilities for each state
        """
        # Method 1: Power iteration
        # Start with uniform distribution
        pi = np.ones(self.n_states) / self.n_states
        
        # Iterate until convergence
        for _ in range(100):
            pi_next = pi @ self.transition_matrix
            if np.allclose(pi, pi_next, rtol=1e-6):
                break
            pi = pi_next
        
        return pi
    
    def calculate_expected_passage_times(self) -> np.ndarray:
        """
        Calculate the expected passage times between all pairs of states.
        
        Returns:
            Matrix of expected passage times where M[i,j] is the expected
            number of steps to go from state i to state j
        """
        n = self.n_states
        M = np.zeros((n, n))
        
        # For each target state j
        for j in range(n):
            # Create a modified transition matrix where state j is absorbing
            P_j = self.transition_matrix.copy()
            P_j[j, :] = 0
            P_j[j, j] = 1
            
            # Create the fundamental matrix
            Q = np.delete(P_j, j, axis=0)
            Q = np.delete(Q, j, axis=1)
            I = np.eye(n - 1)
            N = np.linalg.inv(I - Q)
            
            # Calculate expected passage times
            ones = np.ones(n - 1)
            t = N @ ones
            
            # Insert the results into the passage time matrix
            for i in range(n):
                if i == j:
                    M[i, j] = 1 / self.calculate_steady_state()[i]
                else:
                    i_idx = i if i < j else i - 1
                    M[i, j] = t[i_idx]
        
        return M
    
    def calculate_first_passage_time(self, from_idx: int, to_idx: int) -> float:
        """
        Calculate the expected first passage time from one state to another.
        
        Args:
            from_idx: Index of the source state
            to_idx: Index of the target state
            
        Returns:
            Expected number of steps to reach the target state for the first time
        """
        if from_idx == to_idx:
            # For same state, use recurrence time
            return self.calculate_recurrence_time(from_idx)
        
        # Create a modified transition matrix where target state is absorbing
        P = self.transition_matrix.copy()
        P[to_idx, :] = 0
        P[to_idx, to_idx] = 1
        
        # Remove the target state to create Q matrix
        Q = np.delete(P, to_idx, axis=0)
        Q = np.delete(Q, to_idx, axis=1)
        
        # Calculate fundamental matrix N = (I - Q)^-1
        I = np.eye(self.n_states - 1)
        N = np.linalg.inv(I - Q)
        
        # Calculate expected steps
        ones = np.ones(self.n_states - 1)
        t = N @ ones
        
        # Get the expected time from the source state
        from_idx_adjusted = from_idx if from_idx < to_idx else from_idx - 1
        return t[from_idx_adjusted]
    
    def calculate_recurrence_time(self, state_idx: int) -> float:
        """
        Calculate the expected recurrence time for a state.
        
        Args:
            state_idx: Index of the state
            
        Returns:
            Expected number of steps before returning to the state
        """
        # Recurrence time is the reciprocal of the steady-state probability
        steady_state = self.calculate_steady_state()
        return 1 / steady_state[state_idx]
