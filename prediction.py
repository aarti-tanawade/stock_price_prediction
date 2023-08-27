import streamlit as st
import prophet as pr
from datetime import date

import yfinance as yf
from prophet.plot import plot_plotly
from prophet.plot import plot_plotly
from plotly import graph_objects as go

START="2015-01-01"
TODAY=date.today().strftime("%Y-%m-%d")

st.title("Stock Price Prediction")
stocks=("AAPL","GOOG","MSFT","GME")
selected_stock=st.selectbox("select dataset for prediction", stocks)

n_years=st.slider("years of prediction:", 1,4)
period=n_years*365

@st.cache_data
def load_data(ticker):
    data=yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state=st.text("Load data. . .")
data=load_data(selected_stock)
data_load_state.text("Loading data. . .done!")

st.subheader('Raw Data')
st.write(data.tail())

def plot_raw_data():
    fig=go.Figure(go.Scatter(x=data['Date'], y=data['Open'], name='Stock_open'))
    fig=go.Figure(go.Scatter(x=data['Date'], y=data['Close'], name='Stock_open'))

    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

#forecasting 
df_train=data[['Date','Close']]
df_train=df_train.rename(columns={"Date":"ds","Close":"y"})

m=pr.Prophet()
m.fit(df_train)
future=m.make_future_dataframe(periods=period)
forecast=m.predict(future)
st.subheader('forecast data')
st.write(forecast.tail())

st.write('forecast components')
fig1=plot_plotly(m,forecast)
st.plotly_chart(fig1)

st.write('forecast components')
fig2=m.plot_components(forecast)
st.write(fig2)


