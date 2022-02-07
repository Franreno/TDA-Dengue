"""
    Create fictional data to see how the topological structure will be generated
    - south_increase: 
        Cities below certain latitude will have their cases increased by a factor of 2, while cities above
        given latitude will increase by a constant value
    - north_increase:
        Inverse of south_increase.


    The latitude and longitude chosen will be the geodesic point of the Rio de Janeiro state.
    - O ponto geodésico aos pés de Getúlio Vargas, na mesma praça, que representa o ponto exato 
      onde fica o centro do estado do Rio de Janeiro.
    
"""

import numpy as np
import pandas as pd


OUTPUT_PATH = './data/created-data/'
OUTPUT_FOLDERS = ["south_increase/", "north_increase/"]
OUTPUT_TYPES = ["south_increase_1", "south_increase_2",
                "north_increase_1", "north_increase_2"]
CITIES_LAT_LNG = "./data/cities-lat-lng.csv"

#-22.280636, -42.532351
LATITUDE_LIMIT = -22.280636
LONGITUDE_LIMIT = -42.532351

weeks = np.linspace(1, 52, 52, dtype=int)

SEED = 31415926

starting_cases = 0
np.random.seed(SEED)

cols = list(weeks)
cols.insert(0, "Lng")
cols.insert(0, "Lat")
cols.insert(0, "City")


def create_data_with_factor(city: str, latlng: list, factor=2) -> list:

    starting_cases = np.random.randint(low=10, high=100)

    l = [starting_cases*(factor*week) for week in weeks]
    l.insert(0, latlng[1])
    l.insert(0, latlng[0])
    l.insert(0, city)

    return l


def create_data_with_constant(city: str, latlng: list, constant=1) -> list:

    starting_cases = np.random.randint(low=10, high=100)

    l = [starting_cases+(constant + week) for week in weeks]
    l.insert(0, latlng[1])
    l.insert(0, latlng[0])
    l.insert(0, city)

    return l


def south_increase_1(data, constant=1):
    """
        Below limit increase by a factor of 2.
        Above limit increase by a constant value.
    """

    mainList = []

    for city in data['Cities']:
        city_latitude = data[city][0]

        # -23.006 < -22.28
        if(city_latitude <= LATITUDE_LIMIT):
            mainList.append(create_data_with_factor(city, data[city]))
        else:
            mainList.append(create_data_with_constant(city, data[city]))

    df = pd.DataFrame(mainList, columns=cols)
    df.to_csv(OUTPUT_PATH + OUTPUT_FOLDERS[0] + OUTPUT_TYPES[0] + ".csv")


def south_increase_2(data, constant=1):
    pass


def north_increase_1(data, constant=1):
    """
        Above limit increase by a factor of 2.
        Below limit increase by a constant value.
    """

    mainList = []

    for city in data['Cities']:
        city_latitude = data[city][0]

        # -23.006 > -22.28
        if(city_latitude >= LATITUDE_LIMIT):
            mainList.append(create_data_with_factor(city, data[city]))
        else:
            mainList.append(create_data_with_constant(city, data[city]))

    df = pd.DataFrame(mainList, columns=cols)
    df.to_csv(OUTPUT_PATH + OUTPUT_FOLDERS[1] + OUTPUT_TYPES[2] + ".csv")


def north_increase_2(data, constant=1):
    pass


def to_dict(dataframe: pd.DataFrame) -> dict:
    data = {
        'Cities': list(dataframe['City'])
    }
    for city, lat, lng in zip(dataframe["City"], dataframe["Lat"], dataframe["Lng"]):
        data[city] = [lat, lng]

    return data


if __name__ == '__main__':

    citites_csv = pd.read_csv(CITIES_LAT_LNG)

    data = to_dict(citites_csv)

    south_increase_1(data)
    # south_increase_2(data)
    north_increase_1(data)
    # north_increase_2(data)