#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pandas numpy pyarrow gcsfs')


# In[2]:


print('test')


# In[3]:


import pandas as pd

# Defining file path on GCS

file_path = 'gs://flight-analysis-ms-bucket/landing/Combined_Flights_2018.csv'

# Load the data into a DataFrame
df = pd.read_csv(file_path)

# Display the first few rows and summary info to verify
print(df.head())
print(df.info())


# In[4]:


# Statistical summaries of numerical columns
print(df.describe())


# In[5]:


# Checking for missing values
print(df.isnull().sum())


# In[6]:


# Removing rows with missing values
df_dropped = df.dropna()


# In[7]:


# Finding the mean for the numerical data

# Replace the missing values in ArrDel15 with the mean
df['ArrDel15'] = df['ArrDel15'].fillna(df['ArrDel15'].mean())

#Replace the missing values in ArrivalDelayGroups with median
df['ArrivalDelayGroups'] = df['ArrivalDelayGroups'].fillna(df['ArrivalDelayGroups'].median())

# Replacing the single missing value in DivAirportLandings with 0
df['DivAirportLandings'] = df['DivAirportLandings'].fillna(0)

# Checking for the missing values after imputation
print('Missing values after imputation:')
print(df[['ArrDel15', 'ArrivalDelayGroups', 'DivAirportLandings']].isnull().sum())


# In[8]:


# Checking for any remaining missing values
print(df_dropped.isnull().sum())


# In[9]:


# Display the columns of the original DataFrame
print(df.columns)


# In[10]:


# Descriptive statistics for numeric values
numeric_desc = df.describe()

# Descriptive statistics for the date variable
df['FlightDate'] = pd.to_datetime(df['FlightDate'])  # Convert to datetime format
date_desc = df['FlightDate'].describe()

print(numeric_desc)
print(date_desc)


# In[11]:


# Convert 'Cancelled' to numeric (0 or 1)
df['Cancelled'] = df['Cancelled'].astype(int)

# Select only numeric columns including 'Cancelled'
numeric_df = df.select_dtypes(include=['float64', 'int64', 'int'])

# Calculate the correlation matrix
correlation_matrix = numeric_df.corr()

# Display the correlation of all features with the target variable 'Cancelled'
print(correlation_matrix['Cancelled'].sort_values(ascending=False))


# In[12]:


# Dropping unnecessary columns based on weak correlation
columns_to_drop = [
    'ArrDel15', 'OriginAirportID', 'DayOfWeek', 'DestAirportID',
    'DistanceGroup', 'ArrDelayMinutes', 'AirTime', 'Year'
]

# Drop the columns from the DataFrame
df_cleaned = df.drop(columns=columns_to_drop)

# Check the new shape of the DataFrame
print(df_cleaned.shape)


# In[13]:


# Keep only numeric columns
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Define a correlation threshold (e.g., keep features with |correlation| > 0.01)
correlation_threshold = 0.01

# Get the absolute correlation of each feature with the target variable 'Cancelled'
correlations = abs(numeric_df.corr()['Cancelled'])

# Select features above the threshold
selected_features_corr = correlations[correlations > correlation_threshold].index.tolist()

# Create a DataFrame with only the selected features
df_selected_corr = numeric_df[selected_features_corr]
print("Selected features based on correlation:\n", df_selected_corr.columns)


# In[14]:


# Check for remaining missing values
print("Remaining missing values:\n", df_selected_corr.isnull().sum())


# In[15]:


# Impute missing values for delay-related columns with 0
delay_columns = ['DepDelayMinutes', 'DepDelay', 'DepDel15', 'DepartureDelayGroups']
df_selected_corr.loc[:, delay_columns] = df_selected_corr[delay_columns].fillna(0)

# Impute CRSElapsedTime with the mean
df_selected_corr.loc[:, 'CRSElapsedTime'] = df_selected_corr['CRSElapsedTime'].fillna(df_selected_corr['CRSElapsedTime'].mean())

# Confirm no missing values remain
print("Remaining missing values after imputation:\n", df_selected_corr.isnull().sum().sum())


# In[17]:


import seaborn as sns
import matplotlib.pyplot as plt

# Increase number of bins and set y-axis to logarithmic scale
sns.histplot(df['DepDelayMinutes'], bins=50, kde=True)
plt.title('Distribution of Departure Delay Minutes')
plt.xlabel('Departure Delay Minutes')
plt.ylabel('Frequency')
plt.yscale('log')
plt.show()


# In[18]:


# Boxplot of DepDelayMinutes
sns.boxplot(y='DepDelayMinutes', data=df)
plt.title('Boxplot of Departure Delay Minutes')
plt.ylabel('Departure Delay Minutes')
plt.show()


# In[19]:


import seaborn as sns
import matplotlib.pyplot as plt

# Histogram for DepDelayMinutes focusing on a relevant range (0 to 300 minutes)
plt.figure(figsize=(10, 6))
sns.histplot(df['DepDelayMinutes'], bins=30, kde=True)
plt.xlim(0, 300)  # Focus on delays up to 300 minutes
plt.title('Distribution of Departure Delay Minutes (0-300 mins)')
plt.xlabel('Departure Delay Minutes')
plt.ylabel('Frequency')
plt.show()


# In[20]:


import seaborn as sns
import matplotlib.pyplot as plt

# Set the size of the plots for better visibility
plt.figure(figsize=(10, 6))

# Histogram for DepDelayMinutes focusing on a relevant range (0 to 300 minutes)
sns.histplot(df['DepDelayMinutes'], bins=50, kde=True)  # More bins for detail
plt.xlim(0, 300)  # Focus on delays up to 300 minutes
plt.title('Distribution of Departure Delay Minutes (0-300 mins)')
plt.xlabel('Departure Delay Minutes')
plt.ylabel('Frequency')

# Show the plot
plt.show()

# Optional: Overlay the KDE only
plt.figure(figsize=(10, 6))
sns.kdeplot(df['DepDelayMinutes'], fill=True)  # Show only the KDE
plt.xlim(0, 300)
plt.title('KDE of Departure Delay Minutes (0-300 mins)')
plt.xlabel('Departure Delay Minutes')
plt.ylabel('Density')

# Show the plot
plt.show()

# Optional: Change y-axis to logarithmic scale
plt.figure(figsize=(10, 6))
sns.histplot(df['DepDelayMinutes'], bins=30, kde=True)
plt.xlim(0, 300)
plt.yscale('log')  # Change y-axis to logarithmic scale
plt.title('Distribution of Departure Delay Minutes (0-300 mins) with Log Scale')
plt.xlabel('Departure Delay Minutes')
plt.ylabel('Log Frequency')

# Show the plot
plt.show()


# In[21]:


# Ensure pandas and pyarrow (for Parquet) are installed
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Load the data if not already loaded
# df = pd.read_csv('path/to/your/dataset.csv')  # Uncomment if you need to reload the data

# Convert columns to appropriate data types
df['FlightDate'] = pd.to_datetime(df['FlightDate'])  # Convert date columns to datetime
df['DepDelayMinutes'] = pd.to_numeric(df['DepDelayMinutes'], errors='coerce')  # Convert delay to numeric, handling errors

# Convert specific columns to categorical
categorical_columns = [
    'Airline', 'Origin', 'Dest', 'Cancelled', 'Diverted', 'DepDel15', 
    'ArrivalDelayGroups', 'DistanceGroup', 'OriginState', 'DestState', 
    'Operating_Airline'
]
df[categorical_columns] = df[categorical_columns].astype('category')

# Optional: Specify data types for integer columns
integer_columns = [
    'Year', 'Quarter', 'Month', 'DayofMonth', 'DayOfWeek', 'Distance'
]
df[integer_columns] = df[integer_columns].astype('int64')


# In[22]:


# Schema definition for further validation (this is optional for documentation purposes)
schema = pa.schema([
    ('FlightDate', pa.timestamp('s')),  # Datetime as timestamp
    ('Airline', pa.string()),  # Categorical as string
    ('Origin', pa.string()),
    ('Dest', pa.string()),
    ('Cancelled', pa.int8()),  # Boolean values or integers for binary categories
    ('Diverted', pa.int8()),
    ('DepDel15', pa.int8()),
    ('ArrivalDelayGroups', pa.int8()),
    ('DistanceGroup', pa.int8()),
    ('OriginState', pa.string()),
    ('DestState', pa.string()),
    ('Operating_Airline', pa.string()),
    ('Year', pa.int16()),
    ('Quarter', pa.int8()),
    ('Month', pa.int8()),
    ('DayofMonth', pa.int8()),
    ('DayOfWeek', pa.int8()),
    ('Distance', pa.int32())
])


# In[24]:


import pyarrow as pa
import pyarrow.parquet as pq
import gcsfs

# Define the GCS path for the cleaned data
output_path = 'gs://flight-analysis-ms-bucket/cleaned/cleaned_flight_data.parquet'

# Initialize Google Cloud Storage filesystem
fs = gcsfs.GCSFileSystem()

# Write the cleaned DataFrame to Parquet format on GCS
table = pa.Table.from_pandas(df, schema=schema)
with fs.open(output_path, 'wb') as f:
    pq.write_table(table, f)

print(f"Cleaned data saved to {output_path}")


# In[ ]:




