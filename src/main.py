from sklearn.preprocessing import normalize
import pandas as pd
import numpy as np
import kmapper as km
from sklearn.decomposition import PCA

YEARS = ['2010', '2011', '2012', '2013', '2014', '2015']

INPUT_PATH = './TDA-Dengue/data/strict-data/'
OUTPUT_PATH = "./TDA-Dengue/mappers/strict-data/"


CREATED_DATA_INPUT_PATH = "./data/created-data/"
CREATED_DATA_INPUT_FOLDERS = ["south_increase/", "north_increase/"]
CREATED_DATA_INPUT_TYPES = ["south_increase_1", "south_increase_2",
                            "north_increase_1", "north_increase_2"]
CREATED_DATA_OUTPUT_PATH = "./mappers/created-data/"


def GenerateVectors(df: pd.DataFrame):
    # p = (x,y,w,z) = (Latitude, Longitude, Quantidade de casos acumulados, Semana).

    VectorList = []
    LabelsList = []
    cityList = df["City"]
    for city in cityList:
        row = df.loc[df['City'] == city]
        CityLat = float(row["Lat"])
        CityLng = float(row['Lng'])

        # Iterar por cada semana
        cummulative = 0
        for i in range(1, 53):
            Vector = []
            # Label = []

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

            # Criar o label para esse vector
            LabelsList.append(f"{city}, Casos: {cummulative}, Semana: {i}")

    return VectorList, np.array(LabelsList)


def main(input_path: str, output_path: str):
    DengueDataFrame = pd.read_csv(input_path)
    VectorList, LabelsList = GenerateVectors(DengueDataFrame)

    NormalizedVectorList = normalize(VectorList, norm='l2')

    mapper = km.KeplerMapper(verbose=2)
    perc_overlap = 0.15
    n_cubes = 10
    while(perc_overlap <= 0.45):

        projected_data = mapper.project(NormalizedVectorList, projection=PCA(
            n_components=4), distance_matrix="euclidean")

        # Build a simplicial complex
        graph = mapper.map(projected_data, NormalizedVectorList,
                           cover=km.Cover(n_cubes=n_cubes, perc_overlap=perc_overlap))

        mapper.visualize(
            graph,
            title=f"Casos de dengue no ano",
            path_html=output_path + f"/overlap={perc_overlap}.html",
            custom_tooltips=LabelsList
        )

        perc_overlap += 0.15


if __name__ == '__main__':

    paths = []
    paths.append(CREATED_DATA_INPUT_PATH + CREATED_DATA_INPUT_FOLDERS[0] + CREATED_DATA_INPUT_TYPES[0] + ".csv")
    paths.append(CREATED_DATA_INPUT_PATH + CREATED_DATA_INPUT_FOLDERS[1] + CREATED_DATA_INPUT_TYPES[2] + ".csv")

    outPaths = []
    outPaths.append(CREATED_DATA_OUTPUT_PATH + CREATED_DATA_INPUT_FOLDERS[0])
    outPaths.append(CREATED_DATA_OUTPUT_PATH + CREATED_DATA_INPUT_FOLDERS[1])


    for path, out in zip(paths,outPaths):
        main(path, out)
