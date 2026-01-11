
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler


st.set_page_config(page_title="HotelPreprocessing",layout="wide")
st.title("🏨 Hotel Booking")

#Load Sample
def get_data():
    data=pd.read_csv('hotel bookings.csv') 
    return data.sample(10000,random_state=42).reset_index(drop=True)

df=get_data()
df_processed=df.copy()

#EDA
st.header("1️⃣ Exploratory Data Analysis (EDA)")
with st.expander("View Data Summary"):
    st.write("Data Shape:",df.shape)
    st.write("Column Names:",df.columns.tolist())
    
    col1,col2=st.columns(2)
    with col1:
        st.write("Missing Values Count:")
        st.write(df.isnull().sum())
    with col2:
        st.write("Data Types (Info):")
        st.write(df.dtypes)

    st.write("Statistical Summary (Describe):")
    st.write(df.describe())

    st.write("Booking: (Canceled vs not Canceled):")
    fig,ax=plt.subplots(figsize=(6,3))
    sns.countplot(x='is_canceled',data=df,palette='viridis')
    st.pyplot(fig)

#Data cleaning
st.header("2️⃣ Data cleaning and handling")
with st.container():
    df_processed['reservation_status_date']=pd.to_datetime(df_processed['reservation_status_date'])
    df_processed['res_year']=df_processed['reservation_status_date'].dt.year
    df_processed['res_month']=df_processed['reservation_status_date'].dt.month
    
    df_processed['children']=df_processed['children'].fillna(0)
    df_processed['country']=df_processed['country'].fillna('Unknown')
    df_processed.drop(['company','agent'],axis=1,inplace=True)
    
    col_a,col_b=st.columns(2)
    with col_a:
        st.info("✅Extracted (Year & month) from reservation date.")
        st.dataframe(df_processed[['reservation_status_date','res_year','res_month']].head(3))
    with col_b:
        st.success("✅Handled missing vals and removed unnecessary columns")

#handle outliers ---
st.header("3️⃣ Handling Outliers")
with st.container():
    Q1=df_processed['adr'].quantile(0.25)
    Q3=df_processed['adr'].quantile(0.75)
    IQR=Q3-Q1
    lower_bound=Q1-1.5*IQR
    upper_bound=Q3+1.5*IQR

    df_no_outliers=df_processed[(df_processed['adr']>=lower_bound)&(df_processed['adr']<=upper_bound)].copy()
    
    c1,c2=st.columns(2)
    with c1:
        st.write("Before Cleaning:")
        fig1,ax1=plt.subplots()
        sns.boxplot(x=df_processed['adr'],color='salmon')
        st.pyplot(fig1)
    with c2:
        st.write("After Cleaning:")
        fig2,ax2=plt.subplots()
        sns.boxplot(x=df_no_outliers['adr'],color='skyblue')
        st.pyplot(fig2)

#Encoding & Scaling
st.header("4️⃣ Final Prep (Encoding & Scaling)")
with st.container():
    # Why Label Encoding? 
    # Because Machine Learning models only understand numbers, not text.
    # We convert categories like 'Hotel Type' into 0 and 1.
    st.subheader("Why Label Encoding?")
    st.write("Machine Learning models require numerical input. Label Encoding converts categorical text data into numbers so the model can process it.")
    
    
    
    le=LabelEncoder()
    cols_to_encode=['hotel','meal','customer_type']
    for col in cols_to_encode:
        df_no_outliers[col]=le.fit_transform(df_no_outliers[col])
    
    #Standard Scaling
    scaler=StandardScaler()
    df_no_outliers[['lead_time','adr']]=scaler.fit_transform(df_no_outliers[['lead_time','adr']])
    
    st.write("cleaned data (Numerical):")
    st.dataframe(df_no_outliers.head(5))

#Correlation matrix
st.header("5️⃣ Correlation Matrix (Heatmap)")
with st.container():
    fig_corr,ax_corr=plt.subplots(figsize=(10,6))
    numeric_df=df_no_outliers.select_dtypes(include=['number'])
    sns.heatmap(numeric_df.corr(),annot=False,cmap='coolwarm',ax=ax_corr)
    st.pyplot(fig_corr)

st.sidebar.success("شهد بطيحة")


