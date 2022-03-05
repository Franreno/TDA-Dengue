from sklearn.preprocessing import normalize
import pandas as pd
import numpy as np
import kmapper as km
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

YEARS = ['2010', '2011', '2012', '2013', '2014', '2015']

INPUT_PATH = './TDA-Dengue/data/strict-data/'
OUTPUT_PATH = "./TDA-Dengue/mappers/strict-data/"


CREATED_DATA_INPUT_PATH = "./data/created-data/"
CREATED_DATA_INPUT_FOLDERS = ["south_increase/", "north_increase/"]
CREATED_DATA_PREFIX_TYPES = ["south", "north"]
CREATED_DATA_INPUT_TYPES = ["_Random.csv", "_SameValue.csv"]

CREATED_DATA_OUTPUT_PATH = "./mappers/created-data/"
CREATED_DATA_OUTPUT_TYPES = ["Random/", "SameValue/"]


CREATED_DATA_PATHS = [
    "./data/created-data/north_increase/factors/north_factor=2.csv",
    "./data/created-data/north_increase/factors/north_factor=4.csv",
    "./data/created-data/north_increase/factors/north_factor=16.csv",
    "./data/created-data/south_increase/factors/south_factor=2.csv",
    "./data/created-data/south_increase/factors/south_factor=4.csv",
    "./data/created-data/south_increase/factors/south_factor=16.csv"
]

CREATED_DATA_OUTPUT_PATHS = [
    "./mappers/created-data/north_increase/Factors/north_factor=2.html",
    "./mappers/created-data/north_increase/Factors/north_factor=4.html",
    "./mappers/created-data/north_increase/Factors/north_factor=16.html",
    "./mappers/created-data/south_increase/Factors/south_factor=2.html",
    "./mappers/created-data/south_increase/Factors/south_factor=4.html",
    "./mappers/created-data/south_increase/Factors/south_factor=16.html"
]



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

    NormalizedVectorList = normalize(VectorList, norm='l2', axis=0)

    mapper = km.KeplerMapper()
    perc_overlap = 0.15
    n_cubes = 10


    print(f"FilePath: {input_path}. OutputPath: {output_path}")

    title_to_use = output_path.split('/')[-1].split('.')[0]
    print(f"Process: {title_to_use}.")


    lens = mapper.fit_transform(
        NormalizedVectorList, 
        projection=PCA(),
    )

    # projected_data = mapper.project(NormalizedVectorList, projection=PCA(
        # n_components=4), distance_matrix="euclidean")

    # Build a simplicial complex
    graph = mapper.map(lens, NormalizedVectorList,
                        cover=km.Cover(n_cubes=n_cubes, perc_overlap=perc_overlap),
                        clusterer= DBSCAN()
                        )


    mapper.visualize(
        graph,
        title=title_to_use,
        path_html=output_path,
        custom_tooltips=LabelsList
    )




def create_paths_and_titles():
    input_paths = []
    # south_random.csv
    input_paths.append(CREATED_DATA_INPUT_PATH +
                       CREATED_DATA_INPUT_FOLDERS[0] + CREATED_DATA_PREFIX_TYPES[0] + CREATED_DATA_INPUT_TYPES[0])
    # south_SameValue.csv
    input_paths.append(CREATED_DATA_INPUT_PATH +
                       CREATED_DATA_INPUT_FOLDERS[0] + CREATED_DATA_PREFIX_TYPES[0] + CREATED_DATA_INPUT_TYPES[1])

    # north_random.csv
    input_paths.append(CREATED_DATA_INPUT_PATH +
                       CREATED_DATA_INPUT_FOLDERS[1] + CREATED_DATA_PREFIX_TYPES[1] + CREATED_DATA_INPUT_TYPES[0])
    # north_SameValue.csv
    input_paths.append(CREATED_DATA_INPUT_PATH +
                       CREATED_DATA_INPUT_FOLDERS[1] + CREATED_DATA_PREFIX_TYPES[1] + CREATED_DATA_INPUT_TYPES[1])

    output_paths = []

    output_paths.append(CREATED_DATA_OUTPUT_PATH +
                        CREATED_DATA_INPUT_FOLDERS[0] + CREATED_DATA_OUTPUT_TYPES[0])
    output_paths.append(CREATED_DATA_OUTPUT_PATH +
                        CREATED_DATA_INPUT_FOLDERS[0] + CREATED_DATA_OUTPUT_TYPES[1])
    output_paths.append(CREATED_DATA_OUTPUT_PATH +
                        CREATED_DATA_INPUT_FOLDERS[1] + CREATED_DATA_OUTPUT_TYPES[0])
    output_paths.append(CREATED_DATA_OUTPUT_PATH +
                        CREATED_DATA_INPUT_FOLDERS[1] + CREATED_DATA_OUTPUT_TYPES[1])

    titles = []

    titles.append("Random starting cases. South increase.")
    titles.append("Starting cases = 10. South increase.")
    titles.append("Random starting cases. North increase.")
    titles.append("Starting cases = 10. North increase.")

    return input_paths, output_paths, titles


if __name__ == '__main__':
    inputs_prefix = './data/strict-data/'
    inputs_sufix = 'DengueData.csv'
    output_prefix = './mappers/strict-data/'
    output_sufix = 'DengueData.html'
    years = ['2010', '2011', '2012', '2013', '2014', '2015']


    inputs = []
    outputs = []
    for year in years:
        inputs.append(inputs_prefix + year + inputs_sufix)
        outputs.append(output_prefix + year + '/' + year + output_sufix)
    
    print(inputs, outputs)



    for path, outPath in zip(inputs, outputs):
        main(path, outPath)
