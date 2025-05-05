import numpy as np
from typing import List, Tuple

class HiddenMarkovModel:
    """
    Hidden Markov Model for analyzing latent user intentions behind website navigation.
    """
    
    def __init__(
        self, 
        hidden_states: List[str], 
        observable_states: List[str],
        transition_matrix: np.ndarray,
        emission_matrix: np.ndarray,
        initial_probs: np.ndarray = None
    ):
        """
        Initialize the Hidden Markov Model.
        
        Args:
            hidden_states: List of hidden state names (user intentions)
            observable_states: List of observable state names (website pages)
            transition_matrix: Transition probability matrix between hidden states
            emission_matrix: Emission probability matrix from hidden to observable states
            initial_probs: Initial state probabilities (optional)
        """
        self.hidden_states = hidden_states
        self.observable_states = observable_states
        self.n_hidden = len(hidden_states)
        self.n_observable = len(observable_states)
        
        self.transition_matrix = transition_matrix
        self.emission_matrix = emission_matrix
        
        if initial_probs is not None:
            self.initial_probs = initial_probs
        else:
            # Default to uniform initial probabilities
            self.initial_probs = np.ones(self.n_hidden) / self.n_hidden
    
    def forward(self, observations: List[int]) -> float:
        """
        Implement the Forward algorithm to calculate the probability of an observation sequence.
        
        Args:
            observations: List of observation indices
            
        Returns:
            Probability of the observation sequence given the model
        """
        T = len(observations)
        alpha = np.zeros((T, self.n_hidden))
        
        # Initialize the first step
        for i in range(self.n_hidden):
            alpha[0, i] = self.initial_probs[i] * self.emission_matrix[i, observations[0]]
        
        # Forward recursion
        for t in range(1, T):
            for j in range(self.n_hidden):
                for i in range(self.n_hidden):
                    alpha[t, j] += alpha[t-1, i] * self.transition_matrix[i, j]
                alpha[t, j] *= self.emission_matrix[j, observations[t]]
        
        # Return the sum of the final alpha values
        return np.sum(alpha[T-1])
    
    def viterbi(self, observations: List[int]) -> List[int]:
        """
        Implement the Viterbi algorithm to find the most likely hidden state sequence.
        
        Args:
            observations: List of observation indices
            
        Returns:
            List of hidden state indices representing the most likely path
        """
        T = len(observations)
        
        # Initialize the viterbi and backpointer matrices
        viterbi = np.zeros((T, self.n_hidden))
        backpointer = np.zeros((T, self.n_hidden), dtype=int)
        
        # Initialize the first step
        for i in range(self.n_hidden):
            viterbi[0, i] = self.initial_probs[i] * self.emission_matrix[i, observations[0]]
        
        # Recursion
        for t in range(1, T):
            for j in range(self.n_hidden):
                # Find the most likely previous state
                probs = viterbi[t-1] * self.transition_matrix[:, j]
                backpointer[t, j] = np.argmax(probs)
                viterbi[t, j] = probs[backpointer[t, j]] * self.emission_matrix[j, observations[t]]
        
        # Termination: find the most likely final state
        best_path = [0] * T
        best_path[T-1] = np.argmax(viterbi[T-1])
        
        # Backtrack to find the best path
        for t in range(T-2, -1, -1):
            best_path[t] = backpointer[t+1, best_path[t+1]]
        
        return best_path
    
    def baum_welch(self, observations_list: List[List[int]], max_iter: int = 100, tol: float = 1e-6) -> None:
        """
        Implement the Baum-Welch algorithm to learn the model parameters.
        
        Args:
            observations_list: List of observation sequences
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
        """
        # This is a simplified implementation
        # In a real-world scenario, you would use a more robust implementation
        # such as the one provided by hmmlearn
        
        for _ in range(max_iter):
            # Store old parameters to check for convergence
            old_transition = self.transition_matrix.copy()
            old_emission = self.emission_matrix.copy()
            
            # Accumulators for the new parameters
            new_transition = np.zeros_like(self.transition_matrix)
            new_emission = np.zeros_like(self.emission_matrix)
            
            # Process each observation sequence
            for observations in observations_list:
                T = len(observations)
                
                # Forward pass
                alpha = np.zeros((T, self.n_hidden))
                for i in range(self.n_hidden):
                    alpha[0, i] = self.initial_probs[i] * self.emission_matrix[i, observations[0]]
                
                for t in range(1, T):
                    for j in range(self.n_hidden):
                        for i in range(self.n_hidden):
                            alpha[t, j] += alpha[t-1, i] * self.transition_matrix[i, j]
                        alpha[t, j] *= self.emission_matrix[j, observations[t]]
                
                # Backward pass
                beta = np.zeros((T, self.n_hidden))
                beta[T-1] = 1.0
                
                for t in range(T-2, -1, -1):
                    for i in range(self.n_hidden):
                        for j in range(self.n_hidden):
                            beta[t, i] += beta[t+1, j] * self.transition_matrix[i, j] * self.emission_matrix[j, observations[t+1]]
                
                # Compute xi and gamma
                xi = np.zeros((T-1, self.n_hidden, self.n_hidden))
                gamma = np.zeros((T, self.n_hidden))
                
                # Calculate sequence probability
                p_obs = np.sum(alpha[T-1])
                
                for t in range(T-1):
                    for i in range(self.n_hidden):
                        gamma[t, i] = alpha[t, i] * beta[t, i] / p_obs
                        for j in range(self.n_hidden):
                            xi[t, i, j] = alpha[t, i] * self.transition_matrix[i, j] * self.emission_matrix[j, observations[t+1]] * beta[t+1, j] / p_obs
                
                # Last step for gamma
                for i in range(self.n_hidden):
                    gamma[T-1, i] = alpha[T-1, i] * beta[T-1, i] / p_obs
                
                # Update transition matrix
                for i in range(self.n_hidden):
                    for j in range(self.n_hidden):
                        numerator = np.sum(xi[:, i, j])
                        denominator = np.sum(gamma[:-1, i])
                        if denominator > 0:
                            new_transition[i, j] += numerator / denominator
                
                # Update emission matrix
                for j in range(self.n_hidden):
                    for k in range(self.n_observable):
                        numerator = np.sum(gamma[[t for t in range(T) if observations[t] == k], j])
                        denominator = np.sum(gamma[:, j])
                        if denominator > 0:
                            new_emission[j, k] += numerator / denominator
            
            # Normalize the new parameters
            for i in range(self.n_hidden):
                if np.sum(new_transition[i]) > 0:
                    new_transition[i] /= np.sum(new_transition[i])
                if np.sum(new_emission[i]) > 0:
                    new_emission[i] /= np.sum(new_emission[i])
            
            # Check for convergence
            if (np.abs(new_transition - old_transition).max() < tol and 
                np.abs(new_emission - old_emission).max() < tol):
                break
            
            # Update the model parameters
            self.transition_matrix = new_transition
            self.emission_matrix = new_emission
