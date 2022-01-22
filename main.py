import pandas as pd
import numpy as np
import kmapper as km

YEARS = ['2010', '2011', '2012', '2013', '2014', '2015']

INPUT_PATH = './RJCovidMapper/data/'
OUTPUT_PATH = "./RJCovidMapper/mappers/"


def GenerateVectors(df: pd.DataFrame) -> list:
    # p = (x,y,w,z) = (Latitude, Longitude, Quantidade de casos acumulados, Semana).

    VectorList = []
    cityList = df["City"]
    for city in cityList:
        row = df.loc[df['City'] == city]
        CityLat = float(row["Lat"])
        CityLng = float(row['Lng'])

        # Iterar por cada semana
        cummulative = 0
        for i in range(1, 53):
            Vector = []

            # Latitude
            Vector.append(CityLat)
            # Longitude
            Vector.append(CityLng)
            # Acumula a quantidade de casos
            cummulative += int(row[str(i)])
            # Adiciona no Vector
            Vector.append(cummulative)
            # Semana
            Vector.append(i)
            # Push esse Vector na lista de Vectors.
            VectorList.append(Vector)

    return VectorList


def main(year: str):
    DengueDataFrame = pd.read_csv(INPUT_PATH + year + "DengueData.csv")
    VectorList = np.array(GenerateVectors(DengueDataFrame))

    
    mapper = km.KeplerMapper(verbose=1)

    project_data = mapper.fit_transform(VectorList)

    graph = mapper.map(project_data, VectorList, cover=km.Cover(n_cubes=10))

    mapper.visualize(graph, path_html=OUTPUT_PATH + year + "KepplerMapper.html")


if __name__ == '__main__':
    main('2010')
