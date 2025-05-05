import numpy as np
from typing import List, Tuple

class QueueingModel:
    """
    M/M/1 Queueing Theory model for analyzing website server performance.
    """
    
    def __init__(self, arrival_rate: float, service_rate: float):
        """
        Initialize the M/M/1 queueing model.
        
        Args:
            arrival_rate: Average number of arrivals per time unit (λ)
            service_rate: Average number of services per time unit (μ)
        """
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        
        # Ensure stability condition
        if arrival_rate >= service_rate:
            raise ValueError("Arrival rate must be less than service rate for a stable queue")
    
    def utilization(self) -> float:
        """
        Calculate the server utilization (ρ).
        
        Returns:
            Server utilization (fraction of time server is busy)
        """
        return self.arrival_rate / self.service_rate
    
    def avg_users_in_system(self) -> float:
        """
        Calculate the average number of users in the system (L).
        
        Returns:
            Average number of users in the system (queue + service)
        """
        rho = self.utilization()
        return rho / (1 - rho)
    
    def avg_users_in_queue(self) -> float:
        """
        Calculate the average number of users in the queue (Lq).
        
        Returns:
            Average number of users waiting in the queue
        """
        rho = self.utilization()
        return (rho ** 2) / (1 - rho)
    
    def avg_time_in_system(self) -> float:
        """
        Calculate the average time a user spends in the system (W).
        
        Returns:
            Average time in the system (queue + service)
        """
        return 1 / (self.service_rate - self.arrival_rate)
    
    def avg_time_in_queue(self) -> float:
        """
        Calculate the average time a user spends in the queue (Wq).
        
        Returns:
            Average waiting time in the queue
        """
        return self.arrival_rate / (self.service_rate * (self.service_rate - self.arrival_rate))
    
    def prob_n_users(self, n: int) -> float:
        """
        Calculate the probability of having n users in the system.
        
        Args:
            n: Number of users
            
        Returns:
            Probability of having exactly n users in the system
        """
        rho = self.utilization()
        return (1 - rho) * (rho ** n)
    
    def prob_wait_exceeds(self, t: float) -> float:
        """
        Calculate the probability that waiting time exceeds t.
        
        Args:
            t: Time threshold
            
        Returns:
            Probability that waiting time exceeds t
        """
        return np.exp(-1 * (self.service_rate - self.arrival_rate) * t)
    
    def simulate(self, num_users: int = 1000) -> Tuple[List[float], List[float]]:
        """
        Simulate the M/M/1 queue for a given number of users.
        
        Args:
            num_users: Number of users to simulate
            
        Returns:
            Tuple of (arrival_times, departure_times)
        """
        # Generate inter-arrival times (exponential with rate lambda)
        inter_arrival_times = np.random.exponential(1/self.arrival_rate, num_users)
        arrival_times = np.cumsum(inter_arrival_times)
        
        # Generate service times (exponential with rate mu)
        service_times = np.random.exponential(1/self.service_rate, num_users)
        
        # Calculate departure times
        departure_times = np.zeros(num_users)
        departure_times[0] = arrival_times[0] + service_times[0]
        
        for i in range(1, num_users):
            # Departure time is max of (previous departure time, current arrival time) + service time
            departure_times[i] = max(departure_times[i-1], arrival_times[i]) + service_times[i]
        
        return arrival_times, departure_times
