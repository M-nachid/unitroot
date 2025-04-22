This Streamlit app is a one‐stop dashboard for exploring the stationarity properties of your time‑series data. Its main capabilities include:

Data Ingestion & Overview

Upload your own CSV/Excel file (or load built‑in sample data).

Preview the first few rows, see row/column counts and total missing‑value summary.

Flexible Variable Selection

Pick any subset (or “Select All”) of columns for testing.

Configurable Unit‑Root Testing

Choose among ADF, Phillips‑Perron (PP), KPSS, or combined tests (ADF + PP, ADF + KPSS).

Specify whether to include an intercept, or intercept + linear trend, or neither.

Select lag‑length automatically via AIC/BIC/HQIC/FPE or enter your own max lag.

Set your preferred significance level (1%, 5%, 10%).

Define how many orders of differencing (I(1), I(2), …) to check.

Automated Process‑Type Classification

For each variable, the app:

Runs your chosen tests on the original series.

If non‑stationary, differences once (and again, if needed) until stationarity is found (up to your max).

Determines the order of integration I(d).

Flags the series as Trend‑Stationary (TS) if stationary in level once you remove a significant trend, or Difference‑Stationary (DS) otherwise.

Comprehensive Results Display

Summary table: Combines test statistics, p‑values, lags used, the integration order, and process type for each variable.

Color‑coded styling: Highlights I(0)/I(1)/I(2) and TS vs DS for immediate visual cues.

CSV export link: Download all results for further analysis.

Interactive Visualizations

Order‑of‑Integration Bar Chart: Quickly see which series are I(0), I(1), or I(2).

Detailed per‑series plots (in expandable panels):

Original series

Detrended series (if TS)

First and second differences

Metric panels: Show test statistics, critical values, trend coefficient and its p‑value, plus notes on null vs alternative hypotheses.

Sample Data Generator

One‑click load of seven synthetic series (stationary, trend‑stationary, random walk, etc.) so you can experiment with and learn how different processes behave under each test.

User Experience Enhancements

Custom CSS for a clean, branded look.

Sidebar controls for all settings, with contextual help.

Loading spinners and informative messages to guide you through each step.

All together, this app lets you rigorously test for unit roots, classify your series as TS or DS, visualize their behavior at each differencing stage, and export the results—all without writing a single line of code.
