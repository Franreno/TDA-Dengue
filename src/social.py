'''
Francisco Reis Nogueira
github: @franreno

Analise topologica de dados utilizando o mapper e dados da dengue, turismo domestico ou internacional

To-Do:
    - Decidir a utilizacao de alguma funcao de projecao
    - Escolher o percentual de overlap e a quantidade de hypercubos
    - Analisar os mappers gerados

'''

from typing import OrderedDict, Union
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import kmapper as km
import os
import unidecode

YEARS = ['2010', '2011', '2012', '2013', '2014', '2015']

# Os paths levam em consideracao a organizacao deste repositorio.

workingDirectory  = os.path.abspath( os.getcwd() )

citiesPath = workingDirectory + '/data/Cities-Data.csv'
socialPath = workingDirectory + '/data/moreInfo.csv' 
denguePath = [workingDirectory + f'/data/strict-data/{year}DengueData.csv' for year in YEARS] 

domesticoOutputPaths = [workingDirectory + f'/mappers/strict-data/{year}/{year}DengueComTurismoDomestico.html' for year in YEARS]
internacionalOutputPaths = [workingDirectory + f'/mappers/strict-data/{year}/{year}DengueComTurismoInternacional.html' for year in YEARS]
ambosOutputPaths = [workingDirectory + f'/mappers/strict-data/{year}/{year}DengueComTurismoDomestico-Internacional.html' for year in YEARS]



#---------------------------------------------------------------------------------------------------------------#


def per100k(n: int, citypop: float) -> float:
    '''
        Calcula a quantidade de casos por 100 mil pessoas de um certo municipio
    '''
    return((n/citypop)*100000)


#---------------------------------------------------------------------------------------------------------------#

def readAndCreateData() -> OrderedDict:
    '''
        Realiza a leitura dos csv e organiza-os em um unico dict onde cada cidade possui seus dados
        Exemplo do dict final:
        ```
            mergedData = {
                'municipioDoRJ1': {
                    '2010': {
                        'PIB': 15000,                   # PIB deste municipio
                        'Populacao': 10000,             # Populacao deste municipio
                        'TurismoDomestico': 1203,       # Quantidade de turistas brasileiros que visitaram a municipio
                        'TurismoInternacional': 3111,   # Quantidade de turistas estrangeiros que visitaram o municipio
                        'Casos': [2,3,1,2,3,...]        # Quantidade de casos por semana. Tamanho da lista: 52 (semanas)
                    },
                    '2011': {
                        'PIB': ...
                    }
                },
                'municipioDoRJ2': {
                    ...
                }
            }

        ```

    '''
    mergedData = OrderedDict()

    citiesData = pd.read_csv(citiesPath) 
    socialData = pd.read_csv(socialPath)
    socialData.rename( columns={'Unnamed: 0': 'City'}, inplace=True )
    dengueData = [ pd.read_csv(year).drop(['Lat', 'Lng'], axis=1) for year in denguePath ]
    
    # Padronizar os nomes das cidades do rio
    citiesData['City'] = citiesData['City'].apply(lambda x: unidecode.unidecode(x.lower()))   
    socialData['City'] = socialData['City'].apply(lambda x: unidecode.unidecode(x.lower()))   
    for index, _ in enumerate(YEARS):
        dengueData[index]['City'] = dengueData[index]['City'].apply(lambda x: unidecode.unidecode(x.lower()))   

    # Substituir os valores nulos como zeros
    socialData['Domésticos'] = socialData['Domésticos'].fillna(0)
    socialData['Internacionais'] = socialData['Internacionais'].fillna(0)
    

    # Popular o dict final com os subdicts
    citiesNameList = citiesData['City']
    for city in citiesNameList:
        mergedData[city] = OrderedDict()
        for year in YEARS:
            mergedData[city][year] = OrderedDict()


    # Preencher para cada cidade os dados de um certo ano
    citiesIndexed = citiesData.set_index('City')
    socialIndexed = socialData.set_index('City')

    # Itera pelos anos
    for index, yearData in enumerate(dengueData):
        yearIndexed = yearData.set_index('City')

        # Itera por todas as cidades daquele ano e adiciona os valores
        for city in citiesNameList:
            mergedData[city][YEARS[index]]['PIB'] = citiesIndexed.loc[city]['PIB']
            mergedData[city][YEARS[index]]['Populacao'] = citiesIndexed.loc[city]['Population']
            mergedData[city][YEARS[index]]['TurismoDomestico'] = socialIndexed.loc[city]['Domésticos']
            mergedData[city][YEARS[index]]['TurismoInternacional'] = socialIndexed.loc[city]['Internacionais']
            mergedData[city][YEARS[index]]['RendaPerCapita'] = socialIndexed.loc[city]["Renda per capita"]
            mergedData[city][YEARS[index]]['Casos'] = list(yearIndexed.loc[city][1:53])
        
    return mergedData

#---------------------------------------------------------------------------------------------------------------#


def createVectors(data: OrderedDict, year: str, analysisType: int) -> Union[np.ndarray, np.ndarray]:
    '''
        Cria a nuvem de pontos, utilizando uma lista de vetores, e a lista de labels.
        Cada vetor pode possuir de 4 a 5 features, dependendo do tipo de analise feita.

        p = (x,y,z,w) = (PIB, Turismo, Casos Ate a semana x, semana x)
        (PIB, Renda, Turismos, Casos Ate a semana x, semana x)


        O tipo de analise pode variar sendo:
            1: Turismo domestico
            2: Turismo Internacional
            3: Ambos
    '''
    VectorList = []
    LabelList = []

    cities = list(data.keys())
    # Cria os vetores iterando por todas as cidades
    for city in cities:

        # Pega o ano a ser analisado
        thisCityDict = data[city][year]
        thisCityPopulation = thisCityDict['Populacao']

        # Variavel para a acumulacao dos casos de dengue ate a semana x
        cummulative = 0
        for i in range(0,52):
            Vector = []
            
            # PIB
            Vector.append( thisCityDict['PIB'] )

            # Renda per capita
            Vector.append( thisCityDict['RendaPerCapita'] )

            # Turismo Domestico
            if(analysisType == 1 or analysisType == 3):
                Vector.append( thisCityDict['TurismoDomestico'] )

            # Turismo Internacional
            if(analysisType == 2 or analysisType == 3):
                Vector.append( thisCityDict['TurismoInternacional'] )

            # Casos acumulados
            cummulative += per100k( int(thisCityDict['Casos'][i]), thisCityPopulation )
            Vector.append( cummulative )

            # Semana
            Vector.append(i+1)

            # Adicionar ao PointCloud
            VectorList.append(Vector)

            # Adicionar a lista de labels

            # Se usou domestico
            if(analysisType == 1):
                label = f"{city}, Casos/100k: {cummulative}, Semana: {i}, TDomestico: {thisCityDict['TurismoDomestico']}, PIB: {thisCityDict['PIB']}, Renda Per Capita: {thisCityDict['RendaPerCapita']}"

            # Se usou internacional
            if(analysisType == 2):
                label = f"{city}, Casos/100k: {cummulative}, Semana: {i}, TInternacional: {thisCityDict['TurismoInternacional']}, PIB: {thisCityDict['PIB']}, Renda Per Capita: {thisCityDict['RendaPerCapita']}"

            # Se usou os dois
            if(analysisType == 3):
                label = f"{city}, Casos/100k: {cummulative}, Semana: {i}, TDomestico: {thisCityDict['TurismoDomestico']}, TInternacional: {thisCityDict['TurismoInternacional']}, PIB: {thisCityDict['PIB']}, Renda Per Capita: {thisCityDict['RendaPerCapita']}"


            LabelList.append(label)


    return np.array(VectorList), np.array(LabelList)

#---------------------------------------------------------------------------------------------------------------#

def getOutputPathFromAnalisysType(year: str, analysisType: int):
    '''
        Gera o path para o output do algoritmo mapper.
        Leva em consideracao o ano de analise e o tipo de analise definidos na funcao main().
    '''
    
    # Define o tipo de analise
    variableToUse = None
    if(analysisType == 1):
        variableToUse = domesticoOutputPaths
    elif(analysisType == 2):
        variableToUse = internacionalOutputPaths
    elif(analysisType == 3):
        variableToUse = ambosOutputPaths
    
    mapperTitle = str()
    outputPath = str()

    # Parser para conferir o ano e o titulo do mapper
    for path in variableToUse:
        splittedPath = path.split('/')
        #Check year
        if(year == splittedPath[len(splittedPath)-2]):
            mapperTitle = splittedPath[-1].split('.')[0]
            outputPath = path
        
    return outputPath, mapperTitle

#---------------------------------------------------------------------------------------------------------------#


def generateMapper(NPC: np.ndarray, Labels: np.ndarray, year: str, analysisType: int, perc_overlap: float, n_cubes: int):
    '''
        Aplica o algoritmo mapper e salva a visualizacao.
    '''
    outPath, title = getOutputPathFromAnalisysType(year, analysisType)

    mapper = km.KeplerMapper(verbose=1)


    # Definicao da funcao de projecao
    # Testar o PCA
    # E testar projecoes nas featues (Turismo, ou Renda)

    lens = mapper.project(
        NPC, 
        projection=PCA(),
        # scaler=StandardScaler()
    )

    # Criar um complexo simplicial
    # Utilizar 'lens' ao inves de 'NPC' caso seja utilizado o mapper.project acima
    graph = mapper.map(lens,
                        cover=km.Cover(n_cubes=n_cubes, perc_overlap=perc_overlap),
                        clusterer= DBSCAN()
                        )

    # Criacao da visualizacao e salva ela nas pastas.
    mapper.visualize(
        graph,
        title=title,
        path_html=outPath,
        custom_tooltips=Labels
    )

#---------------------------------------------------------------------------------------------------------------#

def main():

    __PCA = True

    # Ano da analise
    analisedYear = '2011'
    
    # Tipo da analise
    analisysType = 1 # domestico (1), internacional (2), ambos (3)

    # Porcentagem de overlap
    perc_overlap = 0.2

    # Quantidade de hypercubos
    n_cubes = 15


    # Le e cria os dados
    data = readAndCreateData()

    # Cria a nuvem de pontos
    pointCloud, Labels = createVectors(data, analisedYear, analisysType)
    
    # Normaliza a nuvem de pontos
    normalizedPointCloud = normalize(pointCloud, norm='l2', axis=0)

    if(__PCA == True):
        normalizedPointCloud = StandardScaler().fit_transform(normalizedPointCloud)

    # Aplica o algoritmo mapper e gera uma visualizacao
    generateMapper(normalizedPointCloud, Labels, analisedYear, analisysType, perc_overlap, n_cubes)

#---------------------------------------------------------------------------------------------------------------#

if __name__ == '__main__':
    main()