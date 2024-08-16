'''Model Deployment'''
import pickle as pkl
from datetime import datetime
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.title("Apple_Stock_Exchange_Forecast_by_Group")

st.sidebar.title("Time Series Forecasting")
st.sidebar.subheader("This project forecasts Apple stock prices using Prophet and TES models.")

# Display today's date and time
current_date = datetime.now().strftime("%d-%m-%Y")
current_time = datetime.now().strftime("%H:%M")
st.sidebar.write('----')
st.sidebar.write(f"Today's Date: {current_date}")
st.sidebar.write(f"Current Time: {current_time}")

# Title of the app

st.subheader('Select a Model to Forecast the Price')
# Checkbox for terms and conditions

# Model selection
model_choice = st.selectbox("Select the forecasting model", ["Prophet", "TES"])

# Slider for forecast steps
steps = st.slider("Number of steps to forecast", min_value=1, max_value=30, value=30, step=5)

# Button to start the forecast
if st.button('Start Forecast'):
    if model_choice == "Prophet":
        MODEL = 'prophet.pkl'
    else:
        MODEL = 'ExpSM.pkl'

    # Load the chosen model
    with open(MODEL, 'rb') as f:
        model = pkl.load(f)

    # Generate the forecast
    if model_choice == "Prophet":
        # Reading Data
        data = pd.read_csv('./AAPL.csv')
        data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
        data = data.reset_index()[['Date', 'Close']]
        data.columns=['ds', 'y']

        # Creating Dataframe
        future = model.make_future_dataframe(periods=steps)

        # Making Predictions
        forecast = model.predict(future)

        # Plotting Figure
        fig=go.Figure()
        fig.add_trace(
            go.Scatter(
                x = data['ds'],
                y = data['y'],
                mode='lines',
                name ='Actual'
            )
        )
        fig.add_trace(
            go.Scatter(
                x = forecast['ds'],
                y = forecast['yhat'],
                mode = 'lines',
                name = 'Forecast',
                line=dict(color='firebrick')
            )
        )

        # Updating Layout
        fig.update_layout(
            title = 'Stock Price Forecast using Prophet',
            xaxis_title = 'Date',
            yaxis_title='Price'
        )

        # Displaying Figure
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Reading Data

        data = pd.read_csv('./AAPL.csv')
        data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
        data = data.set_index('Date').rename_axis(None)
        data = data[['Close']]

        forecast = model.forecast(steps=steps)

        forecast_dates = pd.date_range(
            start=data.index[-1] + pd.Timedelta(days=1),
            periods=steps,
            freq='D'
        )

        forecast_df = pd.DataFrame(
            forecast.values,
            index=forecast_dates,
            columns=['Forecast'])

        # #st.write(forecast)
        # fig = px.line(
        #     forecast_values,
        #     x=forecast_values.index,
        #     y='Forecasted Value',
        #     title='Forecasted Stock Prices'
        # )

        # Plotting Figure
        fig=go.Figure()
        fig.add_trace(
            go.Scatter(
                x = data.index,
                y = data['Close'],
                mode='lines',
                name ='Actual'
            )
        )
        fig.add_trace(
            go.Scatter(
                x = forecast_df.index,
                y = forecast_df['Forecast'],
                mode = 'lines',
                name = 'Forecast',
                line=dict(color='firebrick')
            )
        )

        # Updating Layout
        fig.update_layout(
            title = 'Stock Price Forecast using Holts-Winter Seasonal Model',
            xaxis_title = 'Date',
            yaxis_title='Price'
        )

        st.plotly_chart(fig, use_container_width=True)
