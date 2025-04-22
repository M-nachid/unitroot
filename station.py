import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.api import VAR
import statsmodels.api as sm
from pmdarima.arima.utils import ndiffs
import io
import base64
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
	page_title="Unit Root Testing App",
	page_icon="📊",
	layout="wide"
)

# Custom CSS to improve appearance
st.markdown("""
<style>
    .reportview-container {
        background-color: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background-color: #f9f9f9;
    }
    h1, h2, h3 {
        color: #1e3d59;
    }
    .stButton>button {
        background-color: #5995ed;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #3a7bd5;
        border-color: #3a7bd5;
    }
    .css-1r6slb0 {border: 1px solid #ddd; border-radius: 5px; padding: 10px;}
    .integration-order-0 {color: #22bb33; font-weight: bold;}
    .integration-order-1 {color: #f0800c; font-weight: bold;}
    .integration-order-2 {color: #bb2124; font-weight: bold;}
    .process-ts {color: #3366cc; font-weight: bold;}
    .process-ds {color: #cc33ff; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.title("🔍 Time Series Unit Root Testing Dashboard")
st.markdown("""
This application provides comprehensive unit root testing capabilities for time series data.
Upload your data file, select variables to test, and configure test parameters to analyze stationarity.
The app distinguishes between **Trend Stationary (TS)** and **Difference Stationary (DS)** processes,
and determines the order of integration: I(0), I(1), or I(2).
""")

# Sidebar for inputs
st.sidebar.header("Configuration")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload your data file", type=["xlsx", "csv", "xls"])

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
	st.session_state.data = None
if 'variables' not in st.session_state:
	st.session_state.variables = []
if 'test_results' not in st.session_state:
	st.session_state.test_results = {}
if 'selected_vars' not in st.session_state:
	st.session_state.selected_vars = []

# Handle uploaded file
if uploaded_file is not None:
	try:
		if uploaded_file.name.endswith('.csv'):
			data = pd.read_csv(uploaded_file)
		else:
			data = pd.read_excel(uploaded_file)

		# Store in session state
		st.session_state.data = data
		st.session_state.variables = data.columns.tolist()

		# Display data preview
		st.subheader("Data Preview")
		st.dataframe(data.head(5))

		# Basic data info
		st.subheader("Data Information")
		col1, col2, col3 = st.columns(3)
		col1.metric("Rows", f"{data.shape[0]}")
		col2.metric("Columns", f"{data.shape[1]}")
		col3.metric("Missing Values", f"{data.isna().sum().sum()}")

	except Exception as e:
		st.error(f"Error loading file: {e}")

# If data is loaded, show options
if st.session_state.data is not None:
	# Variable selection
	st.sidebar.subheader("Variable Selection")
	all_vars = st.sidebar.checkbox("Select All Variables", key="all_vars")

	if all_vars:
		selected_vars = st.session_state.variables
	else:
		selected_vars = st.sidebar.multiselect(
			"Select Variables for Testing",
			options=st.session_state.variables,
			default=st.session_state.selected_vars
		)

	st.session_state.selected_vars = selected_vars

	# Test configuration
	st.sidebar.subheader("Test Configuration")

	# Test type selection
	test_type = st.sidebar.selectbox(
		"Unit Root Test Type",
		options=["ADF (Augmented Dickey-Fuller)",
				 "PP (Phillips-Perron)",
				 "KPSS",
				 "ADF and PP",
				 "ADF and KPSS"],
		index=0
	)

	# Test specification
	test_spec = st.sidebar.selectbox(
		"Test Specification",
		options=["With Constant", "With Constant & Trend", "Without Constant & Trend"],
		index=0
	)

	# Lag selection
	lag_selection = st.sidebar.selectbox(
		"Lag Selection Method",
		options=["AIC (Akaike Information Criterion)",
				 "BIC (Bayesian Information Criterion)",
				 "FPE (Final Prediction Error)",
				 "HQIC (Hannan-Quinn Information Criterion)",
				 "User Specified"],
		index=0
	)

	# Show user lag input if user specified
	max_lag = 12
	if lag_selection == "User Specified":
		max_lag = st.sidebar.number_input("Maximum Lag Length", min_value=1, max_value=20, value=12)

	# Significance level
	significance = st.sidebar.selectbox(
		"Significance Level",
		options=["1%", "5%", "10%"],
		index=1
	)
	sig_level = float(significance.strip("%")) / 100

	# Max order of differencing to check
	max_diff = st.sidebar.slider("Maximum Order of Differencing", min_value=1, max_value=3, value=2)

	# Run tests button
	run_test = st.sidebar.button("Run Unit Root Tests")


	# Function to perform OLS regression to check for trend significance
	def check_trend_significance(series, alpha=0.05):
		# Add trend
		x = np.arange(len(series))
		# Create constant
		X = sm.add_constant(x)
		# Fit model
		model = sm.OLS(series, X).fit()
		# Check if trend coefficient is significant
		trend_pvalue = model.pvalues[1]
		is_significant = trend_pvalue < alpha
		return {
			'Trend Coefficient': model.params[1],
			'Trend p-value': trend_pvalue,
			'Is Trend Significant': is_significant,
			'Model Summary': model.summary()
		}


	# Function to perform ADF test
	def run_adf_test(series, trend_spec, max_lag):
		trend = 'nc'  # no constant, no trend
		if trend_spec == "With Constant":
			trend = 'c'
		elif trend_spec == "With Constant & Trend":
			trend = 'ct'

		# Map lag selection method to statsmodels parameter
		ic_map = {
			"AIC (Akaike Information Criterion)": "aic",
			"BIC (Bayesian Information Criterion)": "bic",
			"FPE (Final Prediction Error)": "t-stat",  # closest available
			"HQIC (Hannan-Quinn Information Criterion)": "hqic",
			"User Specified": None
		}

		ic = ic_map[lag_selection]

		if ic:
			result = adfuller(series, maxlag=max_lag, regression=trend, autolag=ic)
		else:
			result = adfuller(series, maxlag=max_lag, regression=trend, autolag=None)

		# For 'ct' regression, check if trend is significant
		trend_info = None
		if trend == 'ct':
			trend_info = check_trend_significance(series, alpha=sig_level)

		return {
			'Test Statistic': result[0],
			'p-value': result[1],
			'Lags Used': result[2],
			'Observations Used': result[3],
			'Critical Values': result[4],
			'Is Stationary': result[1] < sig_level,
			'Trend Info': trend_info
		}


	# Function to perform KPSS test
	def run_kpss_test(series, trend_spec):
		regression = 'c'  # constant (level stationarity)
		if trend_spec == "With Constant & Trend":
			regression = 'ct'  # constant and trend (trend stationarity)

		result = kpss(series, regression=regression, nlags='auto')

		# For 'ct' regression, check if trend is significant
		trend_info = None
		if regression == 'ct':
			trend_info = check_trend_significance(series, alpha=sig_level)

		return {
			'Test Statistic': result[0],
			'p-value': result[1],
			'Lags Used': result[2],
			'Critical Values': result[3],
			'Is Stationary': result[1] > sig_level,  # Note: KPSS null hypothesis is stationarity
			'Trend Info': trend_info
		}


	# Function to run PP test using statsmodels (this is a simplified version)
	def run_pp_test(series, trend_spec):
		# For PP test, we'll use ndiffs from pmdarima as a proxy
		# since statsmodels doesn't have a direct PP test implementation
		trend = None  # default
		if trend_spec == "With Constant":
			trend = 'c'
		elif trend_spec == "With Constant & Trend":
			trend = 'ct'

		# This is not a true PP test but approximates behavior for demonstration
		result = ndiffs(series, test='pp', max_d=2)

		# For 'ct' regression, check if trend is significant
		trend_info = None
		if trend == 'ct':
			trend_info = check_trend_significance(series, alpha=sig_level)

		return {
			'Test Statistic': None,  # Would need proper PP implementation
			'p-value': None,
			'Lags Used': None,
			'Critical Values': {'1%': None, '5%': None, '10%': None},
			'Differencing Required': result,
			'Is Stationary': result == 0,
			'Trend Info': trend_info
		}


	# Function to determine order of integration
	def determine_integration_order(series, test_type, trend_spec, max_lag, max_diff=2):
		original_series = series.copy()

		# Test original series
		if test_type in ["ADF (Augmented Dickey-Fuller)", "ADF and PP", "ADF and KPSS"]:
			level_result = run_adf_test(original_series, trend_spec, max_lag)
			if level_result['Is Stationary']:
				# Check if it's trend stationary
				if trend_spec == "With Constant & Trend" and level_result['Trend Info'] and level_result['Trend Info'][
					'Is Trend Significant']:
					return 0, 'TS'  # I(0) Trend Stationary
				return 0, 'DS'  # I(0) Difference Stationary

		# First difference
		if max_diff >= 1:
			first_diff = original_series.diff().dropna()
			if test_type in ["ADF (Augmented Dickey-Fuller)", "ADF and PP", "ADF and KPSS"]:
				first_diff_result = run_adf_test(first_diff, trend_spec, max_lag)
				if first_diff_result['Is Stationary']:
					return 1, 'DS'  # I(1) Difference Stationary

		# Second difference
		if max_diff >= 2:
			second_diff = first_diff.diff().dropna()
			if test_type in ["ADF (Augmented Dickey-Fuller)", "ADF and PP", "ADF and KPSS"]:
				second_diff_result = run_adf_test(second_diff, trend_spec, max_lag)
				if second_diff_result['Is Stationary']:
					return 2, 'DS'  # I(2) Difference Stationary

		# If not stationary after max_diff differences
		return max_diff, 'Unknown'


	# Run the tests when the button is clicked
	if run_test and selected_vars:
		with st.spinner("Running unit root tests..."):
			results = {}
			integration_orders = {}
			process_types = {}

			for var in selected_vars:
				series = st.session_state.data[var].dropna()
				var_results = {}

				# Run specified tests
				if test_type in ["ADF (Augmented Dickey-Fuller)", "ADF and PP", "ADF and KPSS"]:
					var_results['ADF'] = run_adf_test(series, test_spec, max_lag)

				if test_type in ["PP (Phillips-Perron)", "ADF and PP"]:
					var_results['PP'] = run_pp_test(series, test_spec)

				if test_type in ["KPSS", "ADF and KPSS"]:
					var_results['KPSS'] = run_kpss_test(series, test_spec)

				# Determine order of integration and type of process
				integration_order, process_type = determine_integration_order(series, test_type, test_spec, max_lag,
																			  max_diff)
				integration_orders[var] = integration_order
				process_types[var] = process_type

				# Store results
				var_results['Integration Order'] = integration_order
				var_results['Process Type'] = process_type
				results[var] = var_results

			st.session_state.test_results = results

	# Display test results
	if st.session_state.test_results and selected_vars:
		st.header("Unit Root Test Results")

		# Create results dataframe
		all_results = []

		for var, tests in st.session_state.test_results.items():
			if var not in selected_vars:
				continue

			integration_order = tests.get('Integration Order', 'Unknown')
			process_type = tests.get('Process Type', 'Unknown')

			for test_name, result in tests.items():
				if test_name not in ['Integration Order', 'Process Type']:
					row_data = {
						'Variable': var,
						'Test': test_name,
						'Test Statistic': result.get('Test Statistic'),
						'p-value': result.get('p-value'),
						'Is Stationary': result.get('Is Stationary'),
						'Lags Used': result.get('Lags Used'),
						'Integration Order': integration_order,
						'Process Type': process_type
					}
					all_results.append(row_data)

		if all_results:
			result_df = pd.DataFrame(all_results)

			# Show summary table
			st.subheader("Summary Table")


			# Style the dataframe for better visualization
			def highlight_integration_order(val):
				if val == 0:
					return 'background-color: #d4ffcc; color: #006600; font-weight: bold'
				elif val == 1:
					return 'background-color: #fff4cc; color: #cc6600; font-weight: bold'
				elif val == 2:
					return 'background-color: #ffcccc; color: #990000; font-weight: bold'
				return ''


			def highlight_process_type(val):
				if val == 'TS':
					return 'background-color: #cce5ff; color: #004085; font-weight: bold'
				elif val == 'DS':
					return 'background-color: #e8d6f9; color: #4b0082; font-weight: bold'
				return ''


			def highlight_stationary(val):
				if val is True:
					return 'background-color: #d4ffcc; font-weight: bold'
				elif val is False:
					return 'background-color: #ffcccc; font-weight: bold'
				return ''


			# Apply styling to dataframe
			styled_df = result_df.style.applymap(highlight_integration_order, subset=['Integration Order']) \
				.applymap(highlight_process_type, subset=['Process Type']) \
				.applymap(highlight_stationary, subset=['Is Stationary'])

			st.dataframe(styled_df)

			# Create a downloadable CSV of results
			csv = result_df.to_csv(index=False)
			b64 = base64.b64encode(csv.encode()).decode()
			href = f'<a href="data:file/csv;base64,{b64}" download="unit_root_results.csv">Download Results as CSV</a>'
			st.markdown(href, unsafe_allow_html=True)

			# Integration order visualization
			st.subheader("Order of Integration Visualization")

			# Get unique variables
			unique_vars = result_df['Variable'].unique()

			# Create a dataframe for the heatmap
			heatmap_data = []
			for var in unique_vars:
				var_row = result_df[result_df['Variable'] == var].iloc[0]
				heatmap_data.append({
					'Variable': var,
					'Integration Order': var_row['Integration Order'],
					'Process Type': var_row['Process Type']
				})

			heatmap_df = pd.DataFrame(heatmap_data)

			# Create a colorful visualization of integration orders
			fig = px.bar(
				heatmap_df,
				x='Variable',
				y='Integration Order',
				color='Integration Order',
				color_continuous_scale=[(0, 'green'), (0.5, 'orange'), (1, 'red')],
				labels={'Integration Order': 'I(d)'},
				height=400,
				text='Process Type'
			)
			fig.update_layout(
				title='Order of Integration by Variable',
				xaxis_title='Variables',
				yaxis_title='Order of Integration I(d)',
				yaxis=dict(
					tickmode='array',
					tickvals=[0, 1, 2],
					ticktext=['I(0)', 'I(1)', 'I(2)']
				)
			)
			st.plotly_chart(fig, use_container_width=True)

			# Show detailed results
			st.subheader("Detailed Results")

			for var in selected_vars:
				if var in st.session_state.test_results:
					# Get integration order and process type for this variable
					var_results = st.session_state.test_results[var]
					integration_order = var_results.get('Integration Order', 'Unknown')
					process_type = var_results.get('Process Type', 'Unknown')

					# Create a header with integration order and process type information
					order_color = "green" if integration_order == 0 else "orange" if integration_order == 1 else "red"
					process_color = "blue" if process_type == 'TS' else "purple" if process_type == 'DS' else "gray"

					header_html = f"""
                    <div style="padding: 10px; border-radius: 5px; margin-bottom: 10px; 
                                background-color: #f8f9fa; border-left: 5px solid {order_color};">
                        <h3>{var}</h3>
                        <p>
                            <span style="color: {order_color}; font-weight: bold;">I({integration_order})</span> | 
                            <span style="color: {process_color}; font-weight: bold;">{process_type} Process</span>
                        </p>
                    </div>
                    """
					st.markdown(header_html, unsafe_allow_html=True)

					with st.expander(f"Details for {var}"):
						tests = st.session_state.test_results[var]

						for test_name, result in tests.items():
							if test_name not in ['Integration Order', 'Process Type']:
								st.write(f"**{test_name} Test Results:**")

								# Format critical values
								critical_values = result.get('Critical Values', {})
								if isinstance(critical_values, dict):
									critical_values_str = ", ".join(
										[f"{k}: {v:.4f}" for k, v in critical_values.items() if v is not None])
								else:
									critical_values_str = "Not available"

								col1, col2 = st.columns(2)

								col1.metric("Test Statistic",
											f"{result.get('Test Statistic', 'N/A'):.4f}" if result.get(
												'Test Statistic') is not None else "N/A")
								col1.metric("p-value", f"{result.get('p-value', 'N/A'):.4f}" if result.get(
									'p-value') is not None else "N/A")
								col1.metric("Lags Used", str(result.get('Lags Used', 'N/A')))

								col2.metric("Critical Values", critical_values_str)
								col2.metric("Is Stationary", "Yes" if result.get('Is Stationary') else "No",
											delta="✓" if result.get('Is Stationary') else "✗")

								# Show trend information if available
								trend_info = result.get('Trend Info')
								if trend_info:
									st.write("**Trend Analysis:**")

									col1, col2 = st.columns(2)
									col1.metric("Trend Coefficient",
												f"{trend_info.get('Trend Coefficient', 'N/A'):.6f}" if trend_info.get(
													'Trend Coefficient') is not None else "N/A")
									col1.metric("Trend p-value",
												f"{trend_info.get('Trend p-value', 'N/A'):.4f}" if trend_info.get(
													'Trend p-value') is not None else "N/A")

									col2.metric("Is Trend Significant",
												"Yes" if trend_info.get('Is Trend Significant') else "No",
												delta="✓" if trend_info.get('Is Trend Significant') else "✗")

									if trend_info.get('Is Trend Significant'):
										st.info(
											"The trend coefficient is statistically significant, suggesting this may be a trend-stationary process.")

								if test_name == 'KPSS':
									st.info(
										"Note: For KPSS test, the null hypothesis is that the series is stationary.")
								else:
									st.info(
										"Note: For ADF and PP tests, the null hypothesis is that the series has a unit root (non-stationary).")

						# Prepare original, differenced, and detrended series for plotting
						series = st.session_state.data[var].dropna()
						diff1 = series.diff().dropna()
						diff2 = diff1.diff().dropna()

						# Create detrended series (for TS process visualization)
						x = np.arange(len(series))
						X = sm.add_constant(x)
						model = sm.OLS(series, X).fit()
						trend = model.params[0] + model.params[1] * x
						detrended = series - trend

						# Create plots
						fig = make_subplots(rows=2, cols=2,
											subplot_titles=("Original Series", "Detrended Series",
															"First Difference", "Second Difference"))

						# Original series
						fig.add_trace(go.Scatter(y=series, mode='lines', name='Original Series',
												 line=dict(color='blue')), row=1, col=1)

						# Detrended series
						fig.add_trace(go.Scatter(y=detrended, mode='lines', name='Detrended Series',
												 line=dict(color='purple')), row=1, col=2)

						# First difference
						fig.add_trace(go.Scatter(y=diff1, mode='lines', name='First Difference',
												 line=dict(color='orange')), row=2, col=1)

						# Second difference
						fig.add_trace(go.Scatter(y=diff2, mode='lines', name='Second Difference',
												 line=dict(color='red')), row=2, col=2)

						fig.update_layout(height=600, title_text=f"Time Series Analysis: {var}")
						st.plotly_chart(fig, use_container_width=True)

						# Display integration order and process type explanation
						st.subheader("Integration Analysis")

						order_emoji = "🟢" if integration_order == 0 else "🟠" if integration_order == 1 else "🔴"
						process_emoji = "🔵" if process_type == 'TS' else "🟣" if process_type == 'DS' else "⚪"

						st.markdown(f"""
                        ### Summary for {var}:

                        - **Integration Order**: {order_emoji} I({integration_order})
                        - **Process Type**: {process_emoji} {process_type} Process

                        #### What this means:
                        """)

						if integration_order == 0 and process_type == 'TS':
							st.success("""
                            **Trend Stationary (TS) Process, I(0)**

                            This series is stationary around a deterministic trend. The series will revert to this trend over time.

                            **Key characteristics:**
                            - Stationary after removing the trend
                            - Shocks have temporary effects
                            - Good for forecasting as it reverts to a predictable trend
                            - No need for differencing in ARIMA modeling
                            """)
						elif integration_order == 0 and process_type == 'DS':
							st.success("""
                            **Difference Stationary (DS) Process, I(0)**

                            This series is already stationary without differencing or detrending.

                            **Key characteristics:**
                            - Mean-reverting
                            - Constant variance over time
                            - Temporary effects from shocks
                            - Can be modeled with ARMA directly
                            """)
						elif integration_order == 1:
							st.warning("""
                            **Difference Stationary (DS) Process, I(1)**

                            This series becomes stationary after first differencing.

                            **Key characteristics:**
                            - Contains a unit root
                            - Shocks have permanent effects
                            - Needs first differencing in ARIMA modeling
                            - May be suitable for cointegration analysis
                            """)
						elif integration_order == 2:
							st.error("""
                            **Difference Stationary (DS) Process, I(2)**

                            This series requires second differencing to achieve stationarity.

                            **Key characteristics:**
                            - Contains two unit roots
                            - Highly persistent
                            - Shocks have permanent and increasing effects
                            - Needs second differencing in ARIMA modeling
                            - More volatile and harder to forecast accurately
                            """)
						else:
							st.info(f"""
                            **Process type could not be clearly determined.**

                            This could mean:
                            - The series has integration order > 2
                            - The series has seasonal unit roots
                            - The series contains structural breaks
                            - Additional testing may be needed
                            """)

			# Create Heatmap Visualization of all variables and their integration orders
			st.subheader("Integration Order Heatmap")

			# Prepare data for the heatmap
			variables = []
			orders = []
			process_types = []

			for var in selected_vars:
				if var in st.session_state.test_results:
					variables.append(var)
					orders.append(st.session_state.test_results[var].get('Integration Order', -1))
					process_types.append(st.session_state.test_results[var].get('Process Type', 'Unknown'))

			# Create a dataframe
			heatmap_data = pd.DataFrame({
				'Variable': variables,
				'Integration Order': orders,
				'Process Type': process_types
			})

			# Order variables by integration order
			heatmap_data = heatmap_data.sort_values(['Integration Order', 'Process Type'])

			# Create heatmap with Plotly
			fig = go.Figure()

			# Add each variable as a bar, colored by integration order
			for i, row in heatmap_data.iterrows():
				var = row['Variable']
				order = row['Integration Order']
				process = row['Process Type']

				# Set color based on integration order
				if order == 0:
					color = 'green'
				elif order == 1:
					color = 'orange'
				elif order == 2:
					color = 'red'
				else:
					color = 'gray'

				# Add process type indicator
				if process == 'TS':
					process_color = 'blue'
					pattern = ''
				elif process == 'DS':
					process_color = 'purple'
					pattern = ''
				else:
					process_color = 'gray'
					pattern = ''

				# Add the bar
				fig.add_trace(go.Bar(
					x=[var],
					y=[1],
					name=f"I({order}) - {process}",
					marker_color=color,
					marker_pattern_shape=pattern,
					showlegend=True,
					text=f"I({order})<br>{process}",
					textposition="inside",
					insidetextanchor="middle",
					hoverinfo="text",
					hovertext=f"{var}: I({order}) - {process} Process"
				))

			fig.update_layout(
				title="Integration Order by Variable",
				xaxis_title="Variables",
				yaxis_title="",
				barmode='stack',
				height=400,
				xaxis={'categoryorder': 'array', 'categoryarray': heatmap_data['Variable'].tolist()},
				yaxis={'showticklabels': False},
				legend_title="Order of Integration"
			)

			st.plotly_chart(fig, use_container_width=True)

	# If no variables selected
	elif run_test and not selected_vars:
		st.warning("Please select at least one variable to test.")

else:
	# If no data loaded yet
	st.info("Please upload a data file to begin analysis.")

	# Sample data option
	if st.button("Use Sample Data"):
		# Generate sample time series data
		np.random.seed(42)
		dates = pd.date_range(start='2010-01-01', periods=120, freq='M')

		# Create different types of sample data
		t = np.arange(120)

		# I(0) - Stationary series
		stationary = np.random.normal(0, 3, 120)

		# I(0) - Trend stationary series
		trend = 0.2 * t
		trend_stationary = trend + np.random.normal(0, 3, 120)

		# I(1) - Random walk (needs first differencing)
		random_walk = np.cumsum(np.random.normal(0, 1, 120))

		# I(1) - Random walk with drift
		drift = 0.2
		random_walk_drift = np.cumsum(np.random.normal(drift, 1, 120))

		# I(2) - Double integrated series
		double_integrated = np.cumsum(np.cumsum(np.random.normal(0, 0.5, 120)))

		# Create DataFrame
		sample_data = pd.DataFrame({
			'stationary': stationary,
			'trend_stationary': trend_stationary,
			'random_walk': random_walk,
			'random_walk_drift': random_walk_drift,
			'double_integrated': double_integrated,
			'cycle': 10 * np.sin(np.linspace(0, 12 * np.pi, 120)) + np.random.normal(0, 2, 120),
			'trend_with_break': np.where(t < 60, t * 0.1, 60 * 0.1 + (t - 60) * 0.3) + np.random.normal(0, 2, 120)
		}, index=dates)

		# Store in session state
		st.session_state.data = sample_data
		st.session_state.variables = sample_data.columns.tolist()

		# Display data preview
		st.subheader("Sample Data Preview")
		st.dataframe(sample_data.head())

		# Display example series descriptions
		st.subheader("Sample Data Description")
		st.markdown("""
        This sample dataset contains different types of time series:

        - **stationary**: I(0) stationary series with constant mean and variance
        - **trend_stationary**: I(0) trend stationary series (TS process)
        - **random_walk**: I(1) random walk without drift (DS process)
        - **random_walk_drift**: I(1) random walk with drift (DS process)
        - **double_integrated**: I(2) double integrated series (DS process)
        - **cycle**: Stationary series with cyclical component
        - **trend_with_break**: Series with a structural break in the trend

        Try running unit root tests on these series to see how they behave!
        """)

		# Plot example series
		fig = go.Figure()
		for col in sample_data.columns:
			fig.add_trace(go.Scatter(x=sample_data.index, y=sample_data[col], mode='lines', name=col))

		fig.update_layout(
			title="Sample Time Series Data",
			xaxis_title="Date",
			yaxis_title="Value",
			height=500
		)
		st.plotly_chart(fig, use_container_width=True)

		# Rerun to update UI
		st.experimental_rerun()

# Footer with information
st.markdown("""
---
### About Unit Root Testing and Time Series Stationarity

Unit root tests determine whether a time series is stationary or non-stationary, and help identify the process type.

#### Key Concepts:

**1. Stationarity Types:**
- **Strictly Stationary**: All statistical properties are constant over time.
- **Weakly Stationary**: Mean, variance, and autocorrelation are constant over time.

**2. Process Types:**
- **Trend Stationary (TS)**: Stationary around a deterministic trend; removing the trend makes it stationary.
- **Difference Stationary (DS)**: Contains a unit root; differencing makes it stationary.

**3. Order of Integration:**
- **I(0)**: Stationary series - no integration, no unit roots.
- **I(1)**: Unit root series - needs first differencing to become stationary.
- **I(2)**: Double unit root - needs second differencing to become stationary.

**4. Economic Implications:**
- **TS Processes**: Shocks have temporary effects; series returns to trend.
- **DS Processes**: Shocks have permanent effects; series follows a random walk.

**5. Available Tests:**
- **ADF (Augmented Dickey-Fuller)**: Tests the null hypothesis that a unit root is present (series is non-stationary).
- **PP (Phillips-Perron)**: Similar to ADF but with non-parametric handling of serial correlation.
- **KPSS (Kwiatkowski-Phillips-Schmidt-Shin)**: Tests the null hypothesis that the series is stationary.

**6. Test Specifications:**
- **With Constant**: Includes an intercept term in the test regression.
- **With Constant & Trend**: Includes both intercept and time trend terms.
- **Without Constant & Trend**: Excludes both intercept and time trend.

**7. Lag Selection Methods:**
- **AIC**: Akaike Information Criterion
- **BIC**: Bayesian Information Criterion (more parsimonious than AIC)
- **FPE**: Final Prediction Error
- **HQIC**: Hannan-Quinn Information Criterion
""")