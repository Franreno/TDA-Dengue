import pandas as pd

INPUT_PATH = './data/strict-data/2010DengueData.csv'
OUTPUT_PATH = './data/cities-lat-lng.csv'

df = pd.read_csv(INPUT_PATH)

cities = df['City']
lat = df['Lat']
lng = df['Lng']

mainList = []
for c,t,g in zip(cities,lat,lng):
    l = []
    l.append(c)
    l.append(t)
    l.append(g)
    mainList.append(l)

cols = ["City", "Lat", "Lng"]

dff = pd.DataFrame(mainList, columns=cols)
dff.to_csv(OUTPUT_PATH)