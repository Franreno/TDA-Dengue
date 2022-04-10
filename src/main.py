from sklearn.preprocessing import normalize
import pandas as pd
import numpy as np
import kmapper as km
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE

YEARS = ['2010', '2011', '2012', '2013', '2014', '2015']

additionalDataPath = './data/Cities-Data.csv'

def GenerateVectors(df: pd.DataFrame):
    # p = (x,y,w,t,z) = (Latitude, Longitude, Quantidade de casos acumulados, IDHM, Semana).

    additionalData = pd.read_csv(additionalDataPath)

    VectorList = []
    LabelsList = []
    cityList = df["City"]
    for city in cityList:
        additionalDataFromCity = additionalData.loc[
            additionalData['City'] == city
        ]
        row = df.loc[df['City'] == city]
        CityLat = float(row["Lat"])
        CityLng = float(row['Lng'])
        CityPop = float(additionalDataFromCity['Population'])
        CityIDHM = float(additionalDataFromCity['IDHM'])
        CityPIB = float(additionalDataFromCity['PIB'])


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

            # Population
            # Vector.append(CityPop)
           
            # IDHM
            Vector.append(CityIDHM)

            # PIB
            Vector.append(CityPIB)


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

    # Build a simplicial complex
    graph = mapper.map(lens,
                        cover=km.Cover(n_cubes=n_cubes, perc_overlap=perc_overlap),
                        clusterer= DBSCAN()
                        )


    mapper.visualize(
        graph,
        title=title_to_use,
        path_html=output_path,
        custom_tooltips=LabelsList
    )


if __name__ == '__main__':
    inputs_prefix = './data/strict-data/'
    inputs_sufix = 'DengueData.csv'
    output_prefix = './mappers/strict-data/'
    output_sufix = 'DengueData-IDHM-PIB.html'
    years = ['2010', '2011', '2012', '2013', '2014', '2015']
    


    inputs = []
    outputs = []
    for year in years:
        inputs.append(inputs_prefix + year + inputs_sufix)
        outputs.append(output_prefix + year + '/' + year + output_sufix)
    

    DengueDataFrame = pd.read_csv(inputs[0])
    VectorList, LabelsList = GenerateVectors(DengueDataFrame)

    NormalizedVectorList = normalize(VectorList, norm='l2', axis=0)

    # main(inputs[0],outputs[0])

    for path, outPath in zip(inputs, outputs):
        main(path, outPath)

