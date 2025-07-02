# Package to imports
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import time
from scipy import stats # For Kolmogorov-Smirnov (KS) test in drift detection
from sklearn.ensemble import IsolationForest # A simple anomaly detection model



# --- Split into two rows of Plot
def split_dataframe_by_rows(df):
    """
    Splits a DataFrame into two, where each resulting DataFrame contains
    data based on a two-row interval from the original.
    """
    df_copy1 = pd.DataFrame(columns=df.columns)
    df_copy2 = pd.DataFrame(columns=df.columns)

    for i in range(0, len(df), 4):
        if i < len(df):
            df_copy1 = pd.concat([df_copy1, df.iloc[i:i+2]])
        if i + 2 < len(df):
            df_copy2 = pd.concat([df_copy2, df.iloc[i:i+2]])
    return df_copy1, df_copy2


# ---- Prepare for splitting the DataFrame ---- #
def prepare_dataframe_for_charting(df: pd.DataFrame):
    """
    Prepares a DataFrame by ensuring 'timestamp', 'value', and 'index_val' columns exist.
    'timestamp' is derived from the DataFrame's index.
    'value' is taken from the first numerical column found.
    'index_val' is the original DataFrame index.
    """
    df_prepared = df.reset_index(drop=False).rename(columns={'index': 'index_val'})

    # Convert index_val to datetime for 'timestamp' if not already present
    if 'timestamp' not in df_prepared.columns:
        # If the original index is numerical, create a datetime index starting from an arbitrary date
        if pd.api.types.is_integer_dtype(df_prepared['index_val']):
            df_prepared['timestamp'] = pd.to_datetime(df_prepared['index_val'], unit='s', origin='2025-01-01')
        else:
            # Fallback for non-integer index_val, convert to string and then to datetime
            df_prepared['timestamp'] = pd.to_datetime(df_prepared['index_val'].astype(str))

    # Find the first numerical column to use as 'value'
    numeric_cols = df_prepared.select_dtypes(include=np.number).columns.tolist()
    # Exclude 'index_val' if it's numeric and we don't want to plot it as 'value'
    if 'index_val' in numeric_cols:
        numeric_cols.remove('index_val')

    if len(numeric_cols) > 0:
        if 'value' not in df_prepared.columns or df_prepared['value'].isnull().all():
            df_prepared['value'] = df_prepared[numeric_cols[0]]
    else:
        st.warning("No numeric columns found to plot as 'value'. Please ensure your CSV has numerical data.")
        df_prepared['value'] = 0 # Default to 0 or handle appropriately if no numeric data

    return df_prepared


# --- Chart Creation Function ---
def create_chart(df: pd.DataFrame, current_index: int, chart_type: str) -> alt.Chart:
    """
    Creates an Altair time series chart with either speculated anomalies (for model view)
    or user-labeled points (for active learning view).

    Args:
        df (pd.DataFrame): The full DataFrame containing time series data.
        current_index (int): The index up to which data should be displayed.
        chart_type (str): 'anomaly_detection' to show speculated anomalies,
                          'active_learning' to show user labels.

    Returns:
        alt.Chart: The composed Altair chart.
    """
    # Ensure current_index does not exceed DataFrame length - 1
    plot_df = df.iloc[:min(current_index + 1, len(df))].copy()

    chart_title = ""
    point_color_value = ""
    point_color_field = None
    point_color_scale = None
    point_marker_shape = ""
    point_filter = None
    tooltip_fields = []

    # Base chart - common for both types
    base = alt.Chart(plot_df).encode(
        x=alt.X('timestamp:T', title='Time'),
        y=alt.Y('value:Q', title='Value')
    )

    # Line for the time series - always a simple blue line
    line = base.mark_line(point=False).encode(
        color=alt.value('steelblue'),
        tooltip=['timestamp:T', 'value:Q']
    )

    if chart_type == 'anomaly_detection':
        chart_title = "Time Series 1: Anomaly Detection (Model's Speculation)"
        point_color_value = 'red'
        point_marker_shape = 'circle'
        plot_df['Is_Speculated_Anomaly'] = plot_df['index_val'].apply(
            lambda x: True if x in st.session_state.speculated_anomalies else False
        )
        point_filter = alt.FieldEqualPredicate(field='Is_Speculated_Anomaly', equal=True)
        tooltip_fields = ['timestamp:T', 'value:Q']

    elif chart_type == 'active_learning':
        chart_title = "Time Series 2: Active Learning (User Labels)"
        # 'Effective_Label' is now expected to be pre-calculated in plot_df
        point_color_field = 'Effective_Label:N'
        point_color_scale = alt.Scale(domain=['Normal', 'Anomaly', 'Yellow_Anomaly', 'Unlabeled'],
                                      range=['green', 'red', 'yellow', 'lightgray']) # Added yellow
        point_marker_shape = 'circle'
        # Filter to show 'Normal', 'Anomaly', and 'Yellow_Anomaly' points
        # Only show points that are explicitly labeled or are yellow anomalies.
        # Unlabeled points (lightgray) will not have their markers shown.
        point_filter = alt.FieldOneOfPredicate('Effective_Label', ['Normal', 'Anomaly', 'Yellow_Anomaly'])
        tooltip_fields = ['timestamp:T', 'value:Q', 'Effective_Label:N', 'z_score:Q'] # Added z_score
    else:
        raise ValueError("Invalid chart_type. Must be 'anomaly_detection' or 'active_learning'.")

    # Points for status (either speculated or labeled)
    if chart_type == 'anomaly_detection':
        status_points = base.mark_point(
            shape=point_marker_shape,
            size=150,
            filled=True,
            color=point_color_value
        ).encode(
            tooltip=tooltip_fields
        ).transform_filter(
            point_filter
        )
    
    else: # active_learning
        status_points = base.mark_point(
            shape=point_marker_shape,
            size=150,
            filled=True
        ).encode(
            color=alt.Color(point_color_field, scale=point_color_scale),
            tooltip=tooltip_fields
        ).transform_filter(
            point_filter
        )

    chart_height = 400
    chart_width = 300

    final_chart = (line + status_points).properties(title=chart_title, height=chart_height, width = chart_width).interactive() # Add .interactive() for zooming/panning

    return final_chart


# --- Anomaly Detection Function ---
def detect_anomalies(df_input: pd.DataFrame, contamination_rate: float = 0.05):
    """
    Performs anomaly detection using Isolation Forest and calculates Z-scores for the given DataFrame.
    Updates st.session_state.speculated_anomalies for the first chart (red dots).
    This function *does not* populate yellow_re_evaluation_indices directly based on model predictions.
    That is handled by update_yellow_anomalies_based_on_labels based on user input.

    Args:
        df_input (pd.DataFrame): The DataFrame to process.
        contamination_rate (float): The proportion of outliers in the data set for Isolation Forest.

    Returns:
        pd.DataFrame: The DataFrame with 'z_score' and 'is_anomaly_if' columns added/updated.
                      Always returns a DataFrame.
    """
    # Work on a copy to avoid SettingWithCopyWarning
    df_working = df_input.copy()

    # Ensure 'z_score' and 'is_anomaly_if' columns exist and are initialized
    if 'z_score' not in df_working.columns:
        df_working['z_score'] = np.nan
    if 'is_anomaly_if' not in df_working.columns:
        df_working['is_anomaly_if'] = False

    if df_working.empty or 'value' not in df_working.columns:
        st.session_state.speculated_anomalies.clear()
        st.session_state.dynamic_z_score_threshold = 2.0 # Default if no data
        return df_working # Return the (potentially empty) DataFrame with default columns

    # Check for sufficient data points for IsolationForest
    if len(df_working) < 5:
        st.session_state.speculated_anomalies.clear()
        st.session_state.dynamic_z_score_threshold = 2.0
        # Ensure z_score and is_anomaly_if columns exist even if too few points
        df_working['z_score'] = np.nan
        df_working['is_anomaly_if'] = False
        return df_working # Return the DataFrame with default columns

    # Using IsolationForest for anomaly detection
    model = IsolationForest(random_state=42, contamination=contamination_rate)
    try:
        model.fit(df_working[['value']]) # Fit on the 'value' column
        df_working.loc[:, 'is_anomaly_if'] = model.predict(df_working[['value']]) == -1

        # Calculate Z-scores
        if len(df_working['value']) > 1: # Z-score requires at least 2 points
            df_working.loc[:, 'z_score'] = np.abs(stats.zscore(df_working['value']))
        else:
            df_working.loc[:, 'z_score'] = 0 # No z-score if only one point

        # --- Dynamic Z-score Threshold Calculation (based on IF anomalies) ---
        # This threshold is calculated from model anomalies, but primarily used for context,
        # not for directly populating yellow_re_evaluation_indices in this revised flow.
        anomaly_z_scores = df_working[df_working['is_anomaly_if']]['z_score']
        
        if not anomaly_z_scores.empty:
            dynamic_z_score_threshold = anomaly_z_scores.median()
            st.session_state.dynamic_z_score_threshold = max(0.5, dynamic_z_score_threshold) # Set a floor
        else:
            st.session_state.dynamic_z_score_threshold = 2.0 # Default if no anomalies detected by IF

        # Update speculated anomalies (red dots for Chart 1)
        # This will be based on the *current slice* passed to it.
        st.session_state.speculated_anomalies.clear() 
        isolation_forest_anomalies = df_working[df_working['is_anomaly_if']]['index_val'].tolist()
        st.session_state.speculated_anomalies.update(isolation_forest_anomalies)
        
        return df_working # Return the DataFrame with new columns

    except ValueError as e:
        st.warning(f"Could not run anomaly detection on this slice: {e}. Data might be too uniform or too small.")
        st.session_state.speculated_anomalies.clear()
        st.session_state.dynamic_z_score_threshold = 2.0
        # Ensure z_score and is_anomaly_if columns exist even on error
        df_working['z_score'] = np.nan
        df_working['is_anomaly_if'] = False
        return df_working # Always return a DataFrame


# --- Data Drift Detection Function ---

def detect_drift(baseline_data: pd.Series, current_data: pd.Series, drift_p_value_threshold: float = 0.05):
    """
    Detects data drift using the Kolmogorov-Smirnov (KS) test.

    Args:
        baseline_data (pd.Series): The reference data series.
        current_data (pd.Series): The current data series to compare.
        drift_p_value_threshold (float): The p-value threshold below which drift is detected.

    Returns:
        bool: True if drift is detected, False otherwise.
        float: The p-value from the KS test.
    """
    if baseline_data.empty or current_data.empty:
        return False, 1.0 # No drift if data is missing

    # Perform KS test
    # statistic: KS test statistic
    # pvalue: two-sided p-value
    statistic, p_value = stats.ks_2samp(baseline_data.dropna(), current_data.dropna())

    # Drift is detected if p-value is below the threshold
    return p_value < drift_p_value_threshold, p_value


# --- Helper to calculate Effective Label ---
def calculate_effective_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the 'Effective_Label' column based on user labels and yellow re-evaluation indices.
    This function should be called after st.session_state.yellow_re_evaluation_indices is updated.
    It prioritizes explicit user labels ('Normal', 'Anomaly') over 'Yellow_Anomaly' status.
    """
    df_with_labels = df.copy()
    df_with_labels['Effective_Label'] = df_with_labels.apply(
        lambda row: 'Normal' if st.session_state.labels.get(row['index_val']) == 'Normal' else (
                            'Anomaly' if st.session_state.labels.get(row['index_val']) == 'Anomaly' else (
                            'Yellow_Anomaly' if row['index_val'] in st.session_state.yellow_re_evaluation_indices else 'Unlabeled'
                            )), axis=1
    )
    return df_with_labels

# --- Helper to update yellow anomalies based on user-labeled anomalies ---
def update_yellow_anomalies_based_on_labels(df_full: pd.DataFrame, anomaly_range_window: int = 50):
    """
    Populates st.session_state.yellow_re_evaluation_indices based *ONLY* on user-labeled 'Anomaly' points.
    Clears all existing yellow indices before re-populating.
    Points within the specified window *before* a user-labeled anomaly become yellow,
    unless they are explicitly labeled 'Normal' by the user.

    Args:
        df_full (pd.DataFrame): The full DataFrame (df_copy2_processed) to reference index values.
        anomaly_range_window (int): Number of points before a user-labeled anomaly to highlight in yellow.
    """
    st.session_state.yellow_re_evaluation_indices.clear() # Clear all previous yellow points

    for labeled_idx_val, label in st.session_state.labels.items():
        if label == 'Anomaly':
            # Ensure the labeled point exists in the current df_full
            if labeled_idx_val in df_full['index_val'].values:
                # Find the true integer location (iloc) of this index_val in df_full
                true_idx_loc = df_full.index[df_full['index_val'] == labeled_idx_val][0]

                # Determine the range of points around the anomaly (only before the anomaly)
                start_range_true_idx = max(0, true_idx_loc - anomaly_range_window)
                end_range_true_idx = true_idx_loc # Include the anomaly point itself

                # Get the 'index_val's for points in this range
                points_in_range = df_full.iloc[start_range_true_idx : end_range_true_idx + 1]['index_val'].tolist()
                
                for idx_val_in_range in points_in_range:
                    # Only add to yellow_re_evaluation_indices if it's not manually labeled 'Normal'
                    # User's 'Normal' label overrides the yellow suggestion
                    if st.session_state.labels.get(idx_val_in_range) != 'Normal':
                        st.session_state.yellow_re_evaluation_indices.add(idx_val_in_range)


# --- Streamlit App ---

st.set_page_config(layout="wide")

st.title("CSV Time Series Analysis with Data Drift & Active Learning (Synthetic Data)")

st.write("""
This application demonstrates time series analysis with synthetic data. It will:
1.  Generate two synthetic time series (`df_copy1`, `df_copy2`) from a base pattern.
2.  For `df_copy1`, it will demonstrate **Anomaly Detection** and highlight suspected anomalies.
    It will also check for **Data Drift** by comparing data distributions over time.
3.  For `df_copy2`, it will provide an interface for **Active Learning**, allowing you to label points.
""")
# --- Session State Initialization ---
if 'speculated_anomalies' not in st.session_state:
    st.session_state.speculated_anomalies = set() # Stores index_val of suspected anomalies

if 'labels' not in st.session_state:
    st.session_state.labels = {} # Stores user labels: {index_val: 'Normal' / 'Anomaly'}

if 'df_original_processed' not in st.session_state:
    st.session_state.df_original_processed = None
if 'df_copy1_processed' not in st.session_state:
    st.session_state.df_copy1_processed = None
if 'df_copy2_processed' not in st.session_state:
    st.session_state.df_copy2_processed = None
if 'current_index' not in st.session_state: # Initialize current_index for the new control panel
    st.session_state.current_index = 0
if 'is_running' not in st.session_state: # Initialize is_running for auto-advance
    st.session_state.is_running = False
if 'model_state' not in st.session_state: # New: Track model's state regarding drift
    st.session_state.model_state = 'stable' # 'stable' or 'drift_detected'
if 'label_history' not in st.session_state: # New: Store history of labels
    st.session_state.label_history = pd.DataFrame(columns=['timestamp', 'index_val', 'label_type', 'action_type'])
if 'yellow_re_evaluation_indices' not in st.session_state: # New: Store indices for yellow points
    st.session_state.yellow_re_evaluation_indices = set() # Changed to set for efficiency
if 'dynamic_z_score_threshold' not in st.session_state:
    st.session_state.dynamic_z_score_threshold = 2.0 # Initial default value

# NEW: Initialize separate indices for each plot
if 'current_index_plot1' not in st.session_state:
    st.session_state.current_index_plot1 = 0
if 'current_index_plot2' not in st.session_state:
    st.session_state.current_index_plot2 = 0


#--- CONSTANTS ---
ANOMALY_PROPAGATION_WINDOW = 50 # Number of points before and after a user-labeled anomaly to highlight in yellow

# --- Synthetic Data Generation ---
def generate_synthetic_data(num_points=300):
    """
    Generates synthetic time series data with a sine wave pattern, drift, and some clear anomalies.
    """
    np.random.seed(42)
    time_index = pd.date_range(start='2025-01-01', periods=num_points, freq='H')
    
    # Base pattern: Sine wave with some noise
    base_signal = 10 * np.sin(np.linspace(0, 20, num_points)) + np.random.normal(0, 1, num_points)

    # Introduce a subtle drift in the middle
    drift_start_idx = num_points // 2
    drift_magnitude = 5
    base_signal[drift_start_idx:] += drift_magnitude * (np.linspace(0, 1, num_points - drift_start_idx)**2)

    # Introduce some clear anomalies
    anomalies_indices = np.random.choice(num_points, 5, replace=False)
    for idx in anomalies_indices:
        base_signal[idx] += np.random.choice([-1, 1]) * np.random.uniform(15, 30) # Large spikes

    df = pd.DataFrame({'timestamp': time_index, 'value': base_signal})
    return df


# Generate data once and store in session state
if st.session_state.df_original_processed is None:
    df_original_synthetic = generate_synthetic_data()
    df_copy1_raw, df_copy2_raw = split_dataframe_by_rows(df_original_synthetic)
    
    st.session_state.df_original_processed = prepare_dataframe_for_charting(df_original_synthetic)
    
    # Process df_copy1 and df_copy2 to add z_score and is_anomaly_if initially
    st.session_state.df_copy1_processed = prepare_dataframe_for_charting(df_copy1_raw)
    st.session_state.df_copy1_processed = detect_anomalies(st.session_state.df_copy1_processed)

    st.session_state.df_copy2_processed = prepare_dataframe_for_charting(df_copy2_raw)
    # Important: Run detect_anomalies here for df_copy2 as well to ensure 'z_score' and 'is_anomaly_if'
    # columns exist, which are needed for the boxplot and tooltips on Chart 2.
    # The 'is_anomaly_if' from this call WILL NOT directly color Chart 2,
    # as calculate_effective_labels will override it based on user labels and yellow_re_evaluation_indices.
    st.session_state.df_copy2_processed = detect_anomalies(st.session_state.df_copy2_processed)


# Make local copies for easier reference (these are updated from session state)
df_original_processed = st.session_state.df_original_processed
df_copy1_processed = st.session_state.df_copy1_processed
df_copy2_processed = st.session_state.df_copy2_processed




# --- Setup Sidebar Control Panel ---
def setup_sidebar():
    """
    Sets up the Streamlit sidebar with navigation, run/pause, and reset controls.
    """
    st.sidebar.header("Controls")
 
    # Previous and Next Buttons
    col_prev, col_next = st.sidebar.columns(2)
    with col_prev:
        if st.button("Prev Point"):
            if st.session_state.current_index > 0:
                st.session_state.current_index -= 1
                # Update specific plot indices based on the main index
                st.session_state.current_index_plot1 = min(st.session_state.current_index, len(df_copy1_processed) - 1)
                st.session_state.current_index_plot2 = min(st.session_state.current_index, len(df_copy2_processed) - 1)
            st.session_state.is_running = False # Pause if navigated manually
            st.rerun()
    with col_next:
        if st.button("Next Point"):
            if st.session_state.current_index < len(st.session_state.df_original_processed) - 1:
                st.session_state.current_index += 1
                # Update specific plot indices based on the main index
                st.session_state.current_index_plot1 = min(st.session_state.current_index, len(df_copy1_processed) - 1)
                st.session_state.current_index_plot2 = min(st.session_state.current_index, len(df_copy2_processed) - 1)
            st.session_state.is_running = False # Pause if navigated manually
            st.rerun()

    st.sidebar.markdown("---")

    # Run and Pause Buttons
    col_run, col_pause = st.sidebar.columns(2)
    with col_run:
        if st.button("Run"):
            st.session_state.is_running = True
            st.rerun() # Rerun to start the loop
    with col_pause:
        if st.button("Pause"):
            st.session_state.is_running = False
            st.rerun() # Rerun to stop the loop

    st.sidebar.markdown("---")     

    # Reset Data and Labels Button
    if st.sidebar.button("Reset Data and Labels"):
        for key in ["current_index", "labels", "is_running", "speculated_anomalies",
                    "df_original_processed", "df_copy1_processed", "df_copy2_processed",
                    "selected_point_index", "yellow_re_evaluation_indices", "dynamic_z_score_threshold",
                    "current_index_plot1", "current_index_plot2", "label_history"]: # Reset all relevant state, including new indices
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    st.sidebar.write(f"Current Global Index: **{st.session_state.current_index}**")
    st.sidebar.write(f"Chart 1 Index: **{st.session_state.current_index_plot1}**")
    st.sidebar.write(f"Chart 2 Index: **{st.session_state.current_index_plot2}**")
    st.sidebar.markdown("---")
    st.sidebar.info("Use the controls to navigate the time series data.")

setup_sidebar() # Call the setup function to render the sidebar

st.write("---")

# --- Column layout for side-by-side charts ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Anomaly Detection & Data Drift")
    st.markdown(f"***Total Speculated Anomalies:*** {len(st.session_state.speculated_anomalies)}")
    # --- Anomaly Detection Logic for df_copy1 ---
    # We run anomaly detection on the currently visible data
    # Removed z_score_threshold for general anomaly detection here, keeping it for the active learning side.
    detect_anomalies(df_copy1_processed.iloc[:st.session_state.current_index_plot1 + 1]) 

    # --- Data Drift Detection Logic for df_copy1 ---
    # Define baseline and current window sizes relative to available data
    # Ensure baseline is from the start and current is from the end of the visible data
    baseline_window_size = min(50, len(df_copy1_processed)) # Fixed baseline size from the beginning of df_copy1
    current_window_size = 50 # Fixed window size for current data
 
    if st.session_state.current_index_plot1 + 1 >= baseline_window_size + current_window_size: # Ensure enough data for both windows
        baseline_data = df_copy1_processed['value'].iloc[:baseline_window_size]
        current_data = df_copy1_processed['value'].iloc[st.session_state.current_index_plot1 - current_window_size + 1 : st.session_state.current_index_plot1 + 1]

        is_drifted, p_value = detect_drift(baseline_data, current_data)

        if is_drifted:
            st.warning(f"⚠️ **DATA DRIFT DETECTED!** (KS p-value: {p_value:.4f})")
            st.info("This indicates a significant change in data distribution. Your anomaly detection model might need retraining or re-evaluation.")
        else:
            st.success(f"✅ Data distribution is stable. (KS p-value: {p_value:.4f})")
    else:
        st.info(f"Need at least {baseline_window_size + current_window_size} points for Data Drift")

    # Display Anomaly Detection Chart
    chart1 = create_chart(df_copy1_processed, st.session_state.current_index_plot1, 'anomaly_detection')
    st.altair_chart(chart1, use_container_width=True)



# --- In col2 ---
with col2:
    st.subheader("Active Learning")
    st.write("\n")

    # Calculate 'Effective_Label' for df_copy2_processed
    st.session_state.df_copy2_processed = calculate_effective_labels(st.session_state.df_copy2_processed)

    # Display Active Learning Chart (passing current_index_plot2)
    chart2 = create_chart(st.session_state.df_copy2_processed, st.session_state.current_index_plot2, 'active_learning')
    st.altair_chart(chart2, use_container_width=True)


# --- Label Data Point (for Active Learning) ---
st.write("\n")
st.write("##### Label Data Point (for Active Learning)") 

# The point to label should be based on current_index_plot2 for df_copy2_processed
# Check bounds for point_to_label_index to prevent IndexError
if st.session_state.current_index_plot2 < len(st.session_state.df_copy2_processed):
    point_to_label_index = st.session_state.df_copy2_processed['index_val'].iloc[st.session_state.current_index_plot2]
    # No need for this sidebar write, as current_index_plot2 is already displayed
    # st.sidebar.write(f"Point to Label Value: **{point_to_label_index}**")

    current_label = st.session_state.labels.get(point_to_label_index, 'Unlabeled')
    st.write(f"Current point to label: Index **{point_to_label_index}** (Time: **{st.session_state.df_copy2_processed['timestamp'].iloc[st.session_state.current_index_plot2].strftime('%Y-%m-%d %H:%M')}**) - Current label: **{current_label}**")

    col_label_btns1, col_label_btns2, col_train_update_btn = st.columns(3) 
    with col_label_btns1:
        if st.button("Label as Normal"):
            st.session_state.labels[point_to_label_index] = 'Normal'
            # If labeled as Normal, remove it from yellow_re_evaluation_indices.
            # This makes it 'invisible' as a yellow anomaly, becoming green.
            if point_to_label_index in st.session_state.yellow_re_evaluation_indices:
                st.session_state.yellow_re_evaluation_indices.remove(point_to_label_index)
            # Log this action to label_history
            new_entry = pd.DataFrame([{
                'timestamp': pd.Timestamp.now(),
                'index_val': point_to_label_index,
                'label_type': 'Normal',
                'action_type': 'Domain expertise' # Corrected typo
            }])
            st.session_state.label_history = pd.concat([st.session_state.label_history, new_entry], ignore_index=True)
            st.rerun()

    with col_label_btns2:
        if st.button("Label as Anomaly"):
            st.session_state.labels[point_to_label_index] = 'Anomaly'
            # If manually labeled as Anomaly, it should no longer be a yellow re-evaluation point,
            # as it's now a confirmed anomaly.
            if point_to_label_index in st.session_state.yellow_re_evaluation_indices:
                st.session_state.yellow_re_evaluation_indices.remove(point_to_label_index)
            # Log this action to label_history
            new_entry = pd.DataFrame([{
                'timestamp': pd.Timestamp.now(),
                'index_val': point_to_label_index,
                'label_type': 'Anomaly',
                'action_type': 'Domain expertise' # Corrected typo
            }])
            st.session_state.label_history = pd.concat([st.session_state.label_history, new_entry], ignore_index=True)
            st.rerun()

    with col_train_update_btn:
        if st.button("Train/update Anomaly Model"):
            # This button will clear previous yellow anomalies and highlight new ones
            # based on *manually labeled anomalies* within a specified propagation range.
            
            # 1. Update the yellow_re_evaluation_indices based on current manual 'Anomaly' labels.
            # This function clears and re-populates the yellow set.
            update_yellow_anomalies_based_on_labels(
                st.session_state.df_copy2_processed, # Pass the full DataFrame to reference indices
                anomaly_range_window=ANOMALY_PROPAGATION_WINDOW # Use the defined constant for the range
            )
            
            # 2. Recalculate Effective_Label for display after yellow_re_evaluation_indices are updated.
            # This ensures the chart colors are correct based on new yellow highlights and user overrides.
            st.session_state.df_copy2_processed = calculate_effective_labels(st.session_state.df_copy2_processed)

            st.info(f"Yellow anomalies updated based on your manual 'Anomaly' labels and surrounding points within a range of {ANOMALY_PROPAGATION_WINDOW} points. Existing manual 'Normal' labels will override yellow highlighting.")
            st.rerun()
 
else:
    st.info("No point to label: Active Learning plot reached end of data or data is insufficient.")

st.markdown(f"**Total Labeled Points:** {len(st.session_state.labels)}")
st.markdown(f"**Normal Labels:** {sum(1 for v in st.session_state.labels.values() if v == 'Normal')}")
st.markdown(f"**Anomaly Labels:** {sum(1 for v in st.session_state.labels.values() if v == 'Anomaly')}")

st.write("---")
st.subheader("Labeling History")
st.dataframe(st.session_state.label_history) 
    
# Auto-advance logic for "Run" button 
# Need this one time run the Time-Series plot
if st.session_state.is_running:
    if st.session_state.current_index < len(st.session_state.df_original_processed) - 1:
        time.sleep(0.3) # Control animation speed
        st.session_state.current_index += 1
        # IMPORTANT: Update plot-specific indices here during auto-advance
        st.session_state.current_index_plot1 = min(st.session_state.current_index, len(df_copy1_processed) - 1)
        st.session_state.current_index_plot2 = min(st.session_state.current_index, len(df_copy2_processed) - 1)
        st.rerun() # Rerun to update the plot with the new index
    else:
        st.session_state.is_running = False # Stop when end is reached
        st.info("Reached end of time series.")
        st.rerun() # Rerun to update button state
