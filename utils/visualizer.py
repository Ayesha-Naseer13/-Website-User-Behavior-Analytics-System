import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from typing import List

def plot_transition_matrix(transition_matrix: np.ndarray, states: List[str]) -> go.Figure:
    """
    Plot the transition matrix as a heatmap.
    
    Args:
        transition_matrix: Transition probability matrix
        states: List of state names
        
    Returns:
        Plotly figure object
    """
    fig = px.imshow(
        transition_matrix,
        x=states,
        y=states,
        color_continuous_scale="Blues",
        labels=dict(x="To", y="From", color="Probability"),
        text_auto=True
    )
    
    fig.update_layout(
        title="Transition Probability Matrix",
        xaxis_title="To",
        yaxis_title="From",
        height=500
    )
    
    return fig

def plot_state_diagram(transition_matrix: np.ndarray, states: List[str]) -> go.Figure:
    """
    Plot the state diagram as a directed graph.
    
    Args:
        transition_matrix: Transition probability matrix
        states: List of state names
        
    Returns:
        Plotly figure object
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes
    for i, state in enumerate(states):
        G.add_node(i, name=state)
    
    # Add edges with weights
    for i in range(len(states)):
        for j in range(len(states)):
            if transition_matrix[i, j] > 0.01:  # Only add edges with non-negligible probability
                G.add_edge(i, j, weight=transition_matrix[i, j])
    
    # Use Kamada-Kawai layout for node positions
    pos = nx.kamada_kawai_layout(G)
    
    # Create edge trace
    edge_x = []
    edge_y = []
    edge_text = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        # Add a slight curve to the edges
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        
        # Add edge weight as text
        edge_text.append(f"{edge[2]['weight']:.2f}")
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(states[node])
        
        # Node size based on outgoing edge weights
        size = sum(transition_matrix[node, :]) * 30
        node_size.append(size)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        marker=dict(
            showscale=False,
            color='#1f77b4',
            size=20,
            line_width=2
        )
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title="State Diagram",
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20, l=5, r=5, t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       height=500
                   ))
    
    return fig

def plot_steady_state(steady_state: np.ndarray, states: List[str]) -> go.Figure:
    """
    Plot the steady-state probabilities.
    
    Args:
        steady_state: Array of steady-state probabilities
        states: List of state names
        
    Returns:
        Plotly figure object
    """
    # Create a dataframe for plotting
    df = pd.DataFrame({
        'State': states,
        'Probability': steady_state
    })
    
    # Sort by probability
    df = df.sort_values('Probability', ascending=False)
    
    # Create bar chart
    fig = px.bar(
        df,
        x='State',
        y='Probability',
        color='Probability',
        color_continuous_scale="Blues",
        text_auto=True
    )
    
    fig.update_layout(
        title="Steady-State Probabilities",
        xaxis_title="State",
        yaxis_title="Probability",
        height=400
    )
    
    return fig

def plot_passage_times(passage_times: np.ndarray, states: List[str], 
                      source: str = None, target: str = None,
                      title: str = "Passage Times") -> go.Figure:
    """
    Plot the passage times between states.
    
    Args:
        passage_times: Matrix of passage times
        states: List of state names
        source: Source state name (optional)
        target: Target state name (optional)
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    if source is not None and target is not None:
        # Highlight specific passage time
        source_idx = states.index(source)
        target_idx = states.index(target)
        
        # Create heatmap
        fig = px.imshow(
            passage_times,
            x=states,
            y=states,
            color_continuous_scale="Blues",
            labels=dict(x="To", y="From", color="Steps"),
            text_auto='.2f'
        )
        
        # Add rectangle to highlight the selected cell
        fig.add_shape(
            type="rect",
            x0=target_idx - 0.5,
            y0=source_idx - 0.5,
            x1=target_idx + 0.5,
            y1=source_idx + 0.5,
            line=dict(color="red", width=2),
            fillcolor="rgba(0,0,0,0)"
        )
    else:
        # Create heatmap for all passage times
        fig = px.imshow(
            passage_times,
            x=states,
            y=states,
            color_continuous_scale="Blues",
            labels=dict(x="To", y="From", color="Steps"),
            text_auto='.2f'
        )
    
    fig.update_layout(
        title=title,
        xaxis_title="To",
        yaxis_title="From",
        height=500
    )
    
    return fig

def plot_hmm_path(observed_path: List[str], hidden_path: List[str]) -> go.Figure:
    """
    Visualize the observed path and inferred hidden states.
    
    Args:
        observed_path: List of observed states (pages)
        hidden_path: List of inferred hidden states (intentions)
        
    Returns:
        Plotly figure object
    """
    # Create a dataframe for plotting
    df = pd.DataFrame({
        'Step': list(range(1, len(observed_path) + 1)),
        'Observed': observed_path,
        'Hidden': hidden_path
    })
    
    # Create figure with secondary y-axis
    fig = go.Figure()
    
    # Add traces for observed path
    fig.add_trace(
        go.Scatter(
            x=df['Step'],
            y=df['Observed'],
            mode='lines+markers',
            name='Observed Pages',
            line=dict(color='blue', width=2),
            marker=dict(size=10)
        )
    )
    
    # Add traces for hidden path
    fig.add_trace(
        go.Scatter(
            x=df['Step'],
            y=df['Hidden'],
            mode='lines+markers',
            name='Hidden Intentions',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=10)
        )
    )
    
    # Update layout
    fig.update_layout(
        title="Observed Path and Inferred Hidden States",
        xaxis=dict(
            title="Step",
            tickmode='linear',
            tick0=1,
            dtick=1
        ),
        yaxis=dict(
            title="State",
            categoryorder='array',
            categoryarray=list(set(observed_path + hidden_path))
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=400
    )
    
    return fig

def plot_queue_metrics(arrival_rate: float, service_rate: float) -> go.Figure:
    """
    Plot queue performance metrics for different utilization levels.
    
    Args:
        arrival_rate: Current arrival rate
        service_rate: Current service rate
        
    Returns:
        Plotly figure object
    """
    # Generate data for different utilization levels
    rho_values = np.linspace(0.1, 0.95, 50)
    L_values = rho_values / (1 - rho_values)  # Average users in system
    W_values = 1 / (service_rate * (1 - rho_values))  # Average time in system
    
    # Current utilization
    current_rho = arrival_rate / service_rate
    current_L = current_rho / (1 - current_rho)
    current_W = 1 / (service_rate - arrival_rate)
    
    # Create figure with secondary y-axis
    fig = go.Figure()
    
    # Add traces for L and W
    fig.add_trace(
        go.Scatter(
            x=rho_values,
            y=L_values,
            mode='lines',
            name='Avg Users in System (L)',
            line=dict(color='blue', width=2)
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=rho_values,
            y=W_values,
            mode='lines',
            name='Avg Time in System (W)',
            line=dict(color='green', width=2),
            yaxis="y2"
        )
    )
    
    # Add point for current values
    fig.add_trace(
        go.Scatter(
            x=[current_rho],
            y=[current_L],
            mode='markers',
            name='Current L',
            marker=dict(color='blue', size=12, symbol='star')
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=[current_rho],
            y=[current_W],
            mode='markers',
            name='Current W',
            marker=dict(color='green', size=12, symbol='star'),
            yaxis="y2"
        )
    )
    
    # Add vertical line for current utilization
    fig.add_vline(
        x=current_rho,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Current ρ = {current_rho:.2f}",
        annotation_position="top right"
    )
    
    # Updated axis-title syntax to avoid invalid 'titlefont'
    fig.update_layout(
        yaxis=dict(
            title=dict(
                text="Avg Users in System (L)",
                font=dict(color="blue", size=14, family='Arial')
            ),
            tickfont=dict(color="blue")
        ),
        yaxis2=dict(
            title=dict(
                text="Avg Time in System (W)",
                font=dict(color="green", size=14, family='Arial')
            ),
            tickfont=dict(color="green"),
            anchor="x",
            overlaying="y",
            side="right"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500
    )
    
    return fig
