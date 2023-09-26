from pymongo.mongo_client import MongoClient
import pandas as pd
import json

# Uniform Resource Indentifier
uri = "mongodb+srv://Piyush:cUxjKK4nwQVaHuK4@cluster0.opvarp6.mongodb.net/?retryWrites=true&w=majority"

# Create a new client and connect to the server
client = MongoClient(uri)

# Creating database and collection names
DATABASE_NAME = 'wafer_data'
COLLECTION_NAME = 'wafer'

# Reading our .csv file
df = pd.read_csv(r'C:\Users\p12m9\Documents\Python Coding\PW\Projects\P2)_WaferFaultDectection\notebook\wafer.csv')
df = df.drop('Unnamed: 0', axis=1)

# Convert DataFrame to list of dictionaries with column names as keys
record = df.to_dict(orient='records')

# Dumping the data into the database
client[DATABASE_NAME][COLLECTION_NAME].insert_many(record)