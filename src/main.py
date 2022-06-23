from sklearn.preprocessing import normalize, StandardScaler
import pandas as pd
import numpy as np
import kmapper as km
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn import ensemble

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


def main(input_path: str, output_path: str, perc=0.15, cubes=10):

    __PCA = False

    DengueDataFrame = pd.read_csv(input_path)
    VectorList, LabelsList = GenerateVectors(DengueDataFrame)

    NormalizedVectorList = normalize(VectorList, norm='l2', axis=0)

    if(__PCA == True):
        NormalizedVectorList = StandardScaler().fit_transform(NormalizedVectorList)

    mapper = km.KeplerMapper(verbose=3)
    perc_overlap = perc
    n_cubes = cubes


    print(f"FilePath: {input_path}. OutputPath: {output_path}")

    title_to_use = output_path.split('/')[-1].split('.')[0]
    print(f"Process: {title_to_use}.")


    # lens = mapper.project(
    #     NormalizedVectorList, 
    #     projection=[2,3],
    #     # scaler=StandardScaler()
    # )

    # Build a simplicial complex
    graph = mapper.map(NormalizedVectorList,
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
    output_sufix = 'DengueData-PIB.html'
    years = ['2010', '2011', '2012', '2013', '2014', '2015']
    

    inputs = []
    outputs = []
    for year in years:
        inputs.append(inputs_prefix + year + inputs_sufix)
        outputs.append(output_prefix + year + '/' + year + output_sufix)
    

    
    main(inputs[0], outputs[0], cubes=10, perc=0.15)
    # main(inputs[2], outputs[2], cubes=10)
    # inputs.pop(0)
    # outputs.pop(0)

    # main(inputs[1], outputs[1])

    # for path, outPath in zip(inputs, outputs):
        # main(path, outPath)
        
