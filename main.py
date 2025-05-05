import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import time 
from models.markov_model import MarkovChain
from models.hmm_model import HiddenMarkovModel
from models.queue_model import QueueingModel
from utils.visualizer import (
    plot_transition_matrix,
    plot_state_diagram,
    plot_steady_state,
    plot_passage_times,
    plot_hmm_path,
    plot_queue_metrics
)

# Set page config
st.set_page_config(
    page_title="Website User Behavior Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5; /* Blue */
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #37474F; /* Blue-grey dark */
        margin-bottom: 1rem;
    }
    .card {
        background-color: #1E1E1E; /* Light grey background */
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #E3F2FD; /* Light blue */
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1976D2; /* Primary blue */
    }
    .metric-label {
        font-size: 0.9rem;
        color: #616161; /* Medium grey */
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'sessions' not in st.session_state:
    st.session_state.sessions = []
if 'page_states' not in st.session_state:
    st.session_state.page_states = ['Home', 'Products', 'Cart', 'Checkout', 'Account']
if 'hidden_states' not in st.session_state:
    st.session_state.hidden_states = ['Exploring', 'Comparing', 'Buying']
if 'transition_matrix' not in st.session_state:
    # Default transition matrix
    st.session_state.transition_matrix = np.array([
        [0.3, 0.4, 0.1, 0.0, 0.2],  # Home
        [0.2, 0.3, 0.4, 0.0, 0.1],  # Products
        [0.1, 0.2, 0.2, 0.5, 0.0],  # Cart
        [0.3, 0.1, 0.1, 0.3, 0.2],  # Checkout
        [0.4, 0.3, 0.1, 0.0, 0.2]   # Account
    ])
if 'emission_matrix' not in st.session_state:
    # Default emission matrix for HMM
    st.session_state.emission_matrix = np.array([
        [0.5, 0.3, 0.1, 0.0, 0.1],  # Exploring
        [0.1, 0.6, 0.2, 0.0, 0.1],  # Comparing
        [0.0, 0.1, 0.3, 0.6, 0.0]   # Buying
    ])
if 'hidden_transition_matrix' not in st.session_state:
    # Default hidden state transition matrix
    st.session_state.hidden_transition_matrix = np.array([
        [0.7, 0.2, 0.1],  # Exploring -> Exploring, Comparing, Buying
        [0.2, 0.6, 0.2],  # Comparing -> Exploring, Comparing, Buying
        [0.1, 0.3, 0.6]   # Buying -> Exploring, Comparing, Buying
    ])

# Sidebar navigation
st.sidebar.markdown('<div class="main-header">Analytics Dashboard</div>', unsafe_allow_html=True)
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "User Simulation", "Markov Chain Analysis", "Hidden Markov Model", "Queueing Theory"]
)

# Overview page
if page == "Overview":
    st.markdown('<div class="main-header">Website User Behavior Analytics System</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3>System Overview</h3>
            <p>This analytics dashboard provides comprehensive insights into website user behavior using probabilistic models:</p>
            <ul>
                <li><strong>Markov Chains</strong> - Model and analyze user page transitions</li>
                <li><strong>Hidden Markov Models (HMMs)</strong> - Uncover latent user intentions</li>
                <li><strong>Queueing Theory (M/M/1)</strong> - Evaluate system performance under load</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h3>How to Use This Dashboard</h3>
            <ol>
                <li>Start by simulating user sessions in the <strong>User Simulation</strong> tab</li>
                <li>Analyze page transitions with <strong>Markov Chain Analysis</strong></li>
                <li>Discover hidden user intentions with the <strong>Hidden Markov Model</strong></li>
                <li>Evaluate server performance with <strong>Queueing Theory</strong></li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h3>Key Metrics</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Display some sample metrics
        if len(st.session_state.sessions) > 0:
            # Calculate some basic metrics
            total_sessions = len(st.session_state.sessions)
            avg_session_length = np.mean([len(session) for session in st.session_state.sessions])
            conversion_rate = sum([1 for session in st.session_state.sessions if 'Checkout' in session]) / total_sessions if total_sessions > 0 else 0
            
            metrics = [
                {"label": "Total Sessions", "value": f"{total_sessions}"},
                {"label": "Avg Session Length", "value": f"{avg_session_length:.2f} pages"},
                {"label": "Conversion Rate", "value": f"{conversion_rate:.1%}"}
            ]
            
            for metric in metrics:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{metric['value']}</div>
                    <div class="metric-label">{metric['label']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No session data available. Please generate some sessions in the User Simulation tab.")
        
        # Sample visualization
        if len(st.session_state.sessions) > 0:
            # Create a simple visualization of the most common paths
            flat_sessions = [page for session in st.session_state.sessions for page in session]
            page_counts = pd.Series(flat_sessions).value_counts()
            
            fig = px.pie(
                values=page_counts.values,
                names=page_counts.index,
                title="Page Visit Distribution",
                color_discrete_sequence=px.colors.sequential.Blues_r
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(margin=dict(t=30, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)

# User Simulation page
elif page == "User Simulation":
    st.markdown('<div class="main-header">User Behavior Simulation</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="sub-header">Generate User Sessions</div>', unsafe_allow_html=True)
        
        simulation_type = st.radio(
            "Simulation Type",
            ["Manual Path Creation", "Random Session Generation"]
        )
        
        if simulation_type == "Manual Path Creation":
            # Manual session creation
            st.markdown("Create a user path by selecting pages in sequence:")
            
            new_session = []
            for i in range(10):  # Allow up to 10 steps
                if i == 0 or (i > 0 and len(new_session) > 0):
                    page_options = st.session_state.page_states
                    selected_page = st.selectbox(
                        f"Step {i+1}",
                        options=[""] + page_options,
                        key=f"manual_step_{i}"
                    )
                    if selected_page:
                        new_session.append(selected_page)
                    else:
                        break
            
            if len(new_session) > 0:
                if st.button("Add This Session"):
                    st.session_state.sessions.append(new_session)
                    st.success(f"Added session: {' → '.join(new_session)}")
        
        else:  # Random Session Generation
            # Random session generation
            num_sessions = st.number_input("Number of Sessions to Generate", min_value=1, max_value=100, value=5)
            min_length = st.slider("Minimum Session Length", min_value=1, max_value=10, value=3)
            max_length = st.slider("Maximum Session Length", min_value=min_length, max_value=15, value=7)
            
            if st.button("Generate Random Sessions"):
                markov = MarkovChain(st.session_state.page_states, st.session_state.transition_matrix)
                
                for _ in range(num_sessions):
                    session_length = np.random.randint(min_length, max_length + 1)
                    session = markov.generate_session(session_length)
                    st.session_state.sessions.append(session)
                
                st.success(f"Generated {num_sessions} random sessions")
    
    with col2:
        st.markdown('<div class="sub-header">Current Sessions</div>', unsafe_allow_html=True)
        
        if len(st.session_state.sessions) > 0:
            # Display existing sessions
            sessions_df = pd.DataFrame({
                "Session ID": [f"Session {i+1}" for i in range(len(st.session_state.sessions))],
                "Path": [" → ".join(session) for session in st.session_state.sessions],
                "Length": [len(session) for session in st.session_state.sessions]
            })
            
            st.dataframe(sessions_df, use_container_width=True)
            
            if st.button("Clear All Sessions"):
                st.session_state.sessions = []
                st.success("All sessions cleared")
            
            # Export option
            if st.download_button(
                "Export Sessions as CSV",
                data=sessions_df.to_csv(index=False),
                file_name="session_data.csv",
                mime="text/csv"
            ):
                st.success("Sessions exported successfully")
        else:
            st.info("No sessions available. Create some sessions using the controls on the left.")
    
    # Visualization of sessions
    if len(st.session_state.sessions) > 0:
        st.markdown('<div class="sub-header">Session Visualization</div>', unsafe_allow_html=True)
        
        # Create a Sankey diagram of user flows
        all_paths = []
        for session in st.session_state.sessions:
            for i in range(len(session) - 1):
                all_paths.append((session[i], session[i+1]))
        
        path_counts = pd.DataFrame(all_paths, columns=['source', 'target']).groupby(['source', 'target']).size().reset_index(name='value')
        
        # Create node labels and indices
        unique_pages = list(set(path_counts['source'].tolist() + path_counts['target'].tolist()))
        node_indices = {page: i for i, page in enumerate(unique_pages)}
        
        # Create Sankey data
        sankey_data = dict(
            type='sankey',
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=unique_pages,
                color=["rgba(31, 119, 180, 0.8)"] * len(unique_pages)
            ),
            link=dict(
                source=[node_indices[row['source']] for _, row in path_counts.iterrows()],
                target=[node_indices[row['target']] for _, row in path_counts.iterrows()],
                value=path_counts['value'].tolist()
            )
        )
        
        # Create and display the figure
        fig = go.Figure(data=[sankey_data])
        fig.update_layout(
            title_text="User Navigation Flows",
            font_size=12,
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

# Markov Chain Analysis page
elif page == "Markov Chain Analysis":
    st.markdown('<div class="main-header">Markov Chain Analysis</div>', unsafe_allow_html=True)
    
    # Initialize Markov model
    markov = MarkovChain(st.session_state.page_states, st.session_state.transition_matrix)
    
    # If we have session data, update the transition matrix
    if len(st.session_state.sessions) > 0:
        with st.spinner("Calculating transition probabilities from session data..."):
            markov.fit(st.session_state.sessions)
            st.session_state.transition_matrix = markov.transition_matrix
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="sub-header">Transition Matrix</div>', unsafe_allow_html=True)
        
        # Option to manually adjust the transition matrix
        manual_adjust = st.checkbox("Manually adjust transition probabilities")
        
        if manual_adjust:
            st.markdown("Adjust transition probabilities (each row must sum to 1):")
            
            # Create a form for matrix editing
            transition_matrix = np.copy(st.session_state.transition_matrix)
            
            for i, source in enumerate(st.session_state.page_states):
                st.markdown(f"**From {source} to:**")
                cols = st.columns(len(st.session_state.page_states))
                row_sum = 0
                
                for j, target in enumerate(st.session_state.page_states):
                    with cols[j]:
                        transition_matrix[i, j] = st.number_input(
                            f"{target}",
                            min_value=0.0,
                            max_value=1.0,
                            value=float(transition_matrix[i, j]),
                            format="%.2f",
                            key=f"tm_{i}_{j}"
                        )
                        row_sum += transition_matrix[i, j]
                
                # Check if row sums to 1
                if not np.isclose(row_sum, 1.0, atol=0.01):
                    st.warning(f"Row sum = {row_sum:.2f}. Should be 1.0")
            
            if st.button("Update Transition Matrix"):
                # Normalize rows to ensure they sum to 1
                for i in range(len(transition_matrix)):
                    transition_matrix[i] = transition_matrix[i] / transition_matrix[i].sum()
                
                st.session_state.transition_matrix = transition_matrix
                markov.transition_matrix = transition_matrix
                st.success("Transition matrix updated")
        
        # Display the transition matrix heatmap
        fig = plot_transition_matrix(markov.transition_matrix, st.session_state.page_states)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="sub-header">State Diagram</div>', unsafe_allow_html=True)
        
        # Display the state diagram
        fig = plot_state_diagram(markov.transition_matrix, st.session_state.page_states)
        st.plotly_chart(fig, use_container_width=True)
    
    # Steady-state probabilities
    st.markdown('<div class="sub-header">Steady-State Probabilities</div>', unsafe_allow_html=True)
    
    steady_state = markov.calculate_steady_state()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Plot steady state probabilities
        fig = plot_steady_state(steady_state, st.session_state.page_states)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h3>Interpretation</h3>
            <p>The steady-state probabilities represent the long-term proportion of time a user spends on each page.</p>
            <p>These values can help identify:</p>
            <ul>
                <li>Most frequently visited pages</li>
                <li>Pages that may need optimization</li>
                <li>Expected user distribution across the site</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Passage times analysis
    st.markdown('<div class="sub-header">Passage Time Analysis</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        source_page = st.selectbox("Source Page", st.session_state.page_states, key="passage_source")
    
    with col2:
        target_page = st.selectbox("Target Page", st.session_state.page_states, key="passage_target")
    
    with col3:
        passage_type = st.selectbox(
            "Analysis Type",
            ["Expected Passage Time", "First Passage Time", "Recurrence Time"]
        )
    
    # Calculate and display passage times
    if passage_type == "Expected Passage Time":
        passage_times = markov.calculate_expected_passage_times()
        fig = plot_passage_times(
            passage_times, 
            st.session_state.page_states,
            source_page,
            target_page,
            "Expected Passage Time (steps)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        source_idx = st.session_state.page_states.index(source_page)
        target_idx = st.session_state.page_states.index(target_page)
        
        st.markdown(f"""
        <div class="card">
            <h3>Expected Passage Time</h3>
            <p>The expected number of steps to go from <strong>{source_page}</strong> to <strong>{target_page}</strong> is 
            <span style="font-size: 1.2rem; font-weight: 600; color: #1976D2;">{passage_times[source_idx, target_idx]:.2f} steps</span>.</p>
            <p>This metric helps understand the typical user journey length between pages.</p>
        </div>
        """, unsafe_allow_html=True)
    
    elif passage_type == "First Passage Time":
        first_passage = markov.calculate_first_passage_time(
            st.session_state.page_states.index(source_page),
            st.session_state.page_states.index(target_page)
        )
        
        st.markdown(f"""
        <div class="card">
            <h3>First Passage Time</h3>
            <p>The expected number of steps for a user to reach <strong>{target_page}</strong> from <strong>{source_page}</strong> 
            for the first time is <span style="font-size: 1.2rem; font-weight: 600; color: #1976D2;">{first_passage:.2f} steps</span>.</p>
            <p>This is useful for understanding how quickly users discover specific pages from different entry points.</p>
        </div>
        """, unsafe_allow_html=True)
    
    else:  # Recurrence Time
        if source_page == target_page:
            recurrence_time = markov.calculate_recurrence_time(st.session_state.page_states.index(source_page))
            
            st.markdown(f"""
            <div class="card">
                <h3>Recurrence Time</h3>
                <p>The expected number of steps before a user returns to <strong>{source_page}</strong> after leaving it is 
                <span style="font-size: 1.2rem; font-weight: 600; color: #1976D2;">{recurrence_time:.2f} steps</span>.</p>
                <p>This metric helps understand page stickiness and return frequency.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Recurrence time is only defined for the same source and target page. Please select the same page in both dropdowns.")
    
    # Next state prediction
    st.markdown('<div class="sub-header">Next State Prediction</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        current_page = st.selectbox("Current Page", st.session_state.page_states, key="current_page")
        
        # Calculate next state probabilities
        current_idx = st.session_state.page_states.index(current_page)
        next_probs = markov.transition_matrix[current_idx]
        
        # Display next state probabilities
        next_probs_df = pd.DataFrame({
            "Next Page": st.session_state.page_states,
            "Probability": next_probs
        }).sort_values("Probability", ascending=False)
        
        st.dataframe(next_probs_df, use_container_width=True)
        
        most_likely_next = st.session_state.page_states[np.argmax(next_probs)]
        
        st.markdown(f"""
        <div class="card">
            <h3>Prediction</h3>
            <p>Most likely next page: <span style="font-size: 1.2rem; font-weight: 600; color: #1976D2;">{most_likely_next}</span></p>
            <p>Probability: <span style="font-weight: 600;">{np.max(next_probs):.2f}</span></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Visualization of next state probabilities
        fig = px.bar(
            next_probs_df,
            x="Next Page",
            y="Probability",
            title=f"Next Page Probabilities from {current_page}",
            color="Probability",
            color_continuous_scale="Blues"
        )
        fig.update_layout(xaxis_title="Next Page", yaxis_title="Probability")
        st.plotly_chart(fig, use_container_width=True)

# Hidden Markov Model page
elif page == "Hidden Markov Model":
    st.markdown('<div class="main-header">Hidden Markov Model Analysis</div>', unsafe_allow_html=True)
    
    # Initialize HMM
    hmm = HiddenMarkovModel(
        st.session_state.hidden_states,
        st.session_state.page_states,
        st.session_state.hidden_transition_matrix,
        st.session_state.emission_matrix
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="sub-header">Model Parameters</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h3>Hidden States (User Intentions)</h3>
            <p>The hidden states represent the underlying user intentions that we cannot directly observe:</p>
            <ul>
                <li><strong>Exploring</strong> - User is browsing and discovering content</li>
                <li><strong>Comparing</strong> - User is evaluating options</li>
                <li><strong>Buying</strong> - User has purchase intent</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Display transition matrix for hidden states
        st.markdown("#### Hidden State Transition Matrix")
        st.markdown("Probability of transitioning between hidden states:")
        
        hidden_transition_df = pd.DataFrame(
            st.session_state.hidden_transition_matrix,
            index=st.session_state.hidden_states,
            columns=st.session_state.hidden_states
        )
        
        fig = px.imshow(
            hidden_transition_df,
            text_auto=True,
            color_continuous_scale="Blues",
            title="Hidden State Transitions"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="sub-header">Emission Probabilities</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h3>Emissions (Observable Pages)</h3>
            <p>The emission probabilities represent how likely each hidden state is to produce each observable page visit:</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display emission matrix
        emission_df = pd.DataFrame(
            st.session_state.emission_matrix,
            index=st.session_state.hidden_states,
            columns=st.session_state.page_states
        )
        
        fig = px.imshow(
            emission_df,
            text_auto=True,
            color_continuous_scale="Blues",
            title="Emission Probabilities"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Path analysis with HMM
    st.markdown('<div class="sub-header">Path Analysis with HMM</div>', unsafe_allow_html=True)
    
    if len(st.session_state.sessions) > 0:
        # Let user select a session to analyze
        session_options = [f"Session {i+1}: {' → '.join(session)}" for i, session in enumerate(st.session_state.sessions)]
        selected_session_idx = st.selectbox("Select a session to analyze:", range(len(session_options)), format_func=lambda x: session_options[x])
        
        selected_session = st.session_state.sessions[selected_session_idx]
        
        # Convert session to indices
        session_indices = [st.session_state.page_states.index(page) for page in selected_session]
        
        # Run Viterbi algorithm to find most likely hidden state sequence
        viterbi_path = hmm.viterbi(session_indices)
        hidden_state_path = [st.session_state.hidden_states[idx] for idx in viterbi_path]
        
        # Calculate observation sequence probability using forward algorithm
        sequence_prob = hmm.forward(session_indices)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Visualize the path with hidden states
            fig = plot_hmm_path(selected_session, hidden_state_path)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown(f"""
            <div class="card">
                <h3>Path Analysis Results</h3>
                <p><strong>Observed Path:</strong><br>{' → '.join(selected_session)}</p>
                <p><strong>Most Likely Hidden States:</strong><br>{' → '.join(hidden_state_path)}</p>
                <p><strong>Sequence Probability:</strong> {sequence_prob:.6f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Interpretation
            st.markdown("""
            <div class="card">
                <h3>Interpretation</h3>
                <p>The HMM reveals the likely user intentions behind the observed navigation pattern.</p>
                <p>This can help with:</p>
                <ul>
                    <li>Understanding user goals</li>
                    <li>Personalizing content based on intent</li>
                    <li>Identifying conversion opportunities</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No sessions available. Please generate some sessions in the User Simulation tab.")
    
    # Forward algorithm demonstration
    st.markdown('<div class="sub-header">Forward Algorithm</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h3>Forward Algorithm</h3>
        <p>The Forward Algorithm calculates the probability of an observation sequence given the model parameters.</p>
        <p>It efficiently computes this by considering all possible hidden state sequences that could have generated the observations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Let user create a custom sequence to analyze
    st.markdown("#### Create a custom sequence to analyze:")
    
    custom_sequence = []
    for i in range(5):  # Allow up to 5 steps
        page = st.selectbox(
            f"Step {i+1}",
            options=[""] + st.session_state.page_states,
            key=f"custom_step_{i}_{int(time.time())}"  # Add timestamp to make unique
        )
        if page:
            custom_sequence.append(page)
        else:
            break
    
    if len(custom_sequence) > 0:
        # Convert custom sequence to indices
        custom_indices = [st.session_state.page_states.index(page) for page in custom_sequence]
        
        # Calculate sequence probability
        custom_prob = hmm.forward(custom_indices)
        
        # Run Viterbi algorithm
        custom_viterbi = hmm.viterbi(custom_indices)
        custom_hidden_path = [st.session_state.hidden_states[idx] for idx in custom_viterbi]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Visualize the custom path
            fig = plot_hmm_path(custom_sequence, custom_hidden_path)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown(f"""
            <div class="card">
                <h3>Custom Sequence Analysis</h3>
                <p><strong>Observed Path:</strong><br>{' → '.join(custom_sequence)}</p>
                <p><strong>Most Likely Hidden States:</strong><br>{' → '.join(custom_hidden_path)}</p>
                <p><strong>Sequence Probability:</strong> {custom_prob:.6f}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Please select at least one page to analyze.")

# Queueing Theory page
elif page == "Queueing Theory":
    st.markdown('<div class="main-header">Queueing Theory Analysis (M/M/1)</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h3>M/M/1 Queue Model</h3>
        <p>The M/M/1 queue is a simple model for analyzing system performance under load:</p>
        <ul>
            <li><strong>M</strong> - Markovian (exponential) arrival process</li>
            <li><strong>M</strong> - Markovian (exponential) service times</li>
            <li><strong>1</strong> - Single server</li>
        </ul>
        <p>This model helps evaluate how your website server handles user traffic.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="sub-header">Queue Parameters</div>', unsafe_allow_html=True)
        
        # Input parameters
        arrival_rate = st.slider(
            "Arrival Rate (λ) - users per minute",
            min_value=1.0,
            max_value=100.0,
            value=30.0,
            step=1.0,
            help="Average number of users arriving per minute"
        )
        
        service_rate = st.slider(
            "Service Rate (μ) - users per minute",
            min_value=arrival_rate + 1.0,
            max_value=200.0,
            value=max(arrival_rate + 10.0, 40.0),
            step=1.0,
            help="Average number of users that can be served per minute"
        )
        
        # Initialize queueing model
        queue_model = QueueingModel(arrival_rate, service_rate)
        
        # Calculate metrics
        utilization = queue_model.utilization()
        avg_users_system = queue_model.avg_users_in_system()
        avg_users_queue = queue_model.avg_users_in_queue()
        avg_time_system = queue_model.avg_time_in_system()
        avg_time_queue = queue_model.avg_time_in_queue()
        
        # Display metrics
        st.markdown("#### System Performance Metrics")
        
        metrics = [
            {"label": "Server Utilization (ρ)", "value": f"{utilization:.2f}", "description": "Fraction of time server is busy"},
            {"label": "Avg Users in System (L)", "value": f"{avg_users_system:.2f}", "description": "Average number of users in the system"},
            {"label": "Avg Users in Queue (Lq)", "value": f"{avg_users_queue:.2f}", "description": "Average number of users waiting in queue"},
            {"label": "Avg Time in System (W)", "value": f"{avg_time_system:.2f} min", "description": "Average time a user spends in the system"},
            {"label": "Avg Time in Queue (Wq)", "value": f"{avg_time_queue:.2f} min", "description": "Average time a user spends waiting in queue"}
        ]
        
        for metric in metrics:
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom: 10px;">
                <div class="metric-value">{metric['value']}</div>
                <div class="metric-label">{metric['label']}</div>
                <div style="font-size: 0.8rem; color: #757575;">{metric['description']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="sub-header">Performance Visualization</div>', unsafe_allow_html=True)
        
        # Plot queue metrics
        fig = plot_queue_metrics(arrival_rate, service_rate)
        st.plotly_chart(fig, use_container_width=True)
        
        # System status
        if utilization < 0.5:
            status = "Excellent"
            color = "green"
            description = "The system is handling traffic efficiently with minimal waiting times."
        elif utilization < 0.7:
            status = "Good"
            color = "blue"
            description = "The system is handling traffic well but may experience occasional delays."
        elif utilization < 0.9:
            status = "Warning"
            color = "orange"
            description = "The system is approaching capacity. Users may experience noticeable delays."
        else:
            status = "Critical"
            color = "red"
            description = "The system is near or at capacity. Users will experience significant delays."
        
        st.markdown(f"""
        <div class="card" style="border-left: 5px solid {color};">
            <h3>System Status: <span style="color: {color};">{status}</span></h3>
            <p>{description}</p>
            <p>Recommendations:</p>
            <ul>
                <li>{"Consider scaling up server capacity" if utilization > 0.7 else "Current capacity is sufficient"}</li>
                <li>{"Implement load balancing" if utilization > 0.8 else "Monitor traffic patterns"}</li>
                <li>{"Optimize page load times" if avg_time_system > 1.0 else "Current response times are good"}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # What-if analysis
    st.markdown('<div class="sub-header">What-If Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h3>Traffic Scenario Analysis</h3>
        <p>Explore how different traffic scenarios affect system performance.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate data for different arrival rates
    arrival_rates = np.linspace(1, service_rate * 0.95, 20)
    utilizations = []
    wait_times = []
    queue_lengths = []
    
    for arr_rate in arrival_rates:
        model = QueueingModel(arr_rate, service_rate)
        utilizations.append(model.utilization())
        wait_times.append(model.avg_time_in_system())
        queue_lengths.append(model.avg_users_in_system())
    
    # Create a dataframe for plotting
    scenario_df = pd.DataFrame({
        "Arrival Rate": arrival_rates,
        "Utilization": utilizations,
        "Wait Time": wait_times,
        "Queue Length": queue_lengths
    })
    
    # Plot the scenarios
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=scenario_df["Arrival Rate"],
        y=scenario_df["Utilization"],
        mode="lines",
        name="Utilization",
        line=dict(color="blue", width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=scenario_df["Arrival Rate"],
        y=scenario_df["Wait Time"],
        mode="lines",
        name="Wait Time (min)",
        line=dict(color="green", width=2),
        yaxis="y2"
    ))
    
    # Add vertical line for current arrival rate
    fig.add_vline(
        x=arrival_rate,
        line_dash="dash",
        line_color="red",
        annotation_text="Current",
        annotation_position="top right"
    )
    
    # Update layout for dual y-axis
    # … after adding your traces …

    fig.update_layout(
        title="System Performance vs. Traffic",
        xaxis_title="Arrival Rate (users/min)",
        yaxis=dict(
            title=dict(
                text="Utilization",
                font=dict(color="blue")
            ),
            tickfont=dict(color="blue"),
            range=[0, 1]
        ),
        yaxis2=dict(
            title=dict(
                text="Wait Time (min)",
                font=dict(color="green")
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

    st.plotly_chart(fig, use_container_width=True)

    
    # Maximum capacity analysis
    st.markdown("#### Maximum Capacity Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        target_wait_time = st.slider(
            "Target Maximum Wait Time (minutes)",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.1
        )
    
    with col2:
        target_utilization = st.slider(
            "Target Maximum Utilization",
            min_value=0.5,
            max_value=0.95,
            value=0.8,
            step=0.05
        )
    
    # Calculate maximum arrival rate based on constraints
    max_arrival_rate_util = service_rate * target_utilization
    max_arrival_rate_wait = service_rate - (1 / target_wait_time)
    max_arrival_rate = min(max_arrival_rate_util, max_arrival_rate_wait)
    
    if max_arrival_rate > 0:
        st.markdown(f"""
        <div class="card">
            <h3>Maximum Capacity Results</h3>
            <p>Based on your constraints, the maximum arrival rate your system can handle is 
            <span style="font-size: 1.2rem; font-weight: 600; color: #1976D2;">{max_arrival_rate:.2f} users/minute</span>.</p>
            <p>This is limited by your {"wait time constraint" if max_arrival_rate_wait < max_arrival_rate_util else "utilization constraint"}.</p>
            <p>At this arrival rate:</p>
            <ul>
                <li>Server utilization will be {(max_arrival_rate / service_rate):.2f}</li>
                <li>Average wait time will be {(1 / (service_rate - max_arrival_rate)):.2f} minutes</li>
                <li>Average queue length will be {(max_arrival_rate**2 / (service_rate * (service_rate - max_arrival_rate))):.2f} users</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("The constraints are too strict. Please increase the target wait time or utilization.")

if __name__ == "__main__":
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown("Website User Behavior Analytics System v1.0")
    st.sidebar.markdown("Built with Streamlit and Python")
