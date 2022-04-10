import pandas as pd
import numpy as np

COLS = ['City', 'Lat', 'Lng', 'Population', 'IDHM', 'PIB']

datapath = "../data/cities-lat-lng.csv"
output = "../data/Cities-Data.csv"
fullcitiespath = "../data/fullcitiesdata.xlsx"

dataframe = pd.read_csv(datapath)

fullcitiesDataframe = pd.read_excel(fullcitiespath)

citiesColumns = fullcitiesDataframe.columns
fullcitiesDataframe = fullcitiesDataframe[
    [
        citiesColumns[0], 
        citiesColumns[5], 
        citiesColumns[8],  
        citiesColumns[-1]
    ]
]

fullcitiesDataframe = fullcitiesDataframe.rename(columns={
    citiesColumns[0]: COLS[0],
    citiesColumns[5]: COLS[3],
    citiesColumns[8]: COLS[4],
    citiesColumns[-1]: COLS[5]
})

newDataDict = {}
latlngDict = {}

for index, city in enumerate(fullcitiesDataframe['City']):
    _newData = fullcitiesDataframe.iloc[index]
    newDataDict[city] = {
        'Population': _newData[1],
        'IDHM': _newData[2],
        'PIB': _newData[3]
    }

for index, city in enumerate(dataframe['City']):
    _newData = dataframe.iloc[index]

    latlngDict[city] = {
        'Lat': _newData[2],
        'Lng': _newData[3] 
    }



finalList = []

citiesList = dataframe['City']
for index, city in enumerate(citiesList):
    finalList.append(
        [
            city, 
            latlngDict[city]['Lat'], 
            latlngDict[city]['Lng'],
            newDataDict[city]['Population'],
            newDataDict[city]['IDHM'],
            newDataDict[city]['PIB']
        ]
    )


dff = pd.DataFrame(finalList, columns=COLS)
dff.to_csv(output)