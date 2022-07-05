'''
Francisco Reis Nogueira
github: @franreno

Como utilizar os labels em conjunto com o KeplerMapper:
Para juntar os Labels com o Mapper eh necessario criar uma 
nuvem de pontos juntamente com os labels para cada ponto.

Esse codigo nao gera um grafo compreensivel!
Puramente demonstrativo para como usar as labels.
O grafo gerado sera de apenas um no contendo todos os pontos.
'''

from typing import Tuple
from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN
import numpy as np
import kmapper as km

#-------------------------------------------------------#

# DATASET FICTICIO
ALTURAS = [100,240,120,320,120,320]
CITY = ["C1", "C2", "C3", "C4", "C5", "C6"]
TEMP = [25,30,12,16,21,6]

#-------------------------------------------------------#

def printArraysAndLabels(PC: np.ndarray, Labels: np.ndarray):
    for point, label in zip(PC, Labels):
        print(f"Label: {label}\nPoint: {point}\n")

#-------------------------------------------------------#

def CreateArrays() -> Tuple[np.ndarray,np.ndarray]:
    '''
    Os Arrays devem ser criados em conjuntos
    Um array de informacoes + array de labels

    p = (y,z)
        y: Altura da cidade
        z: Temperatura na cidade

    l = (Cidade, Altura, Temperatura)
    
    '''
    PointCloud = []
    Labels = []

    for name, height, temp in zip(CITY, ALTURAS, TEMP):
        
        # Array com as informações
        newArray = []
        newArray.append( height )
        newArray.append( temp )

        # Adicionar o ponto na nuvem de pontos
        PointCloud.append( newArray )

        # Label para esse array que esta sendo criado:
        # Aqui vc pode colocar qualquer informacao que quiser 
        # O mapper ira linkar esse label com o ponto da nuvem de pontos
        # desde que sejam criados ao mesmo tempo 
        Labels.append( f"{name}, Altura: {height}, Temperatura: {temp}" )

    # Por fim transformar ambos em vetores do numpy.
    return np.array(PointCloud), np.array(Labels)

#-------------------------------------------------------#

def main():
    pointCloud, Labels = CreateArrays()
    printArraysAndLabels(pointCloud, Labels)


    # Normalizar
    nPointCloud = normalize(pointCloud, norm='l2', axis=0)


    # Utilizacao no KepplerMapper
    mapper = km.KeplerMapper()

    # Funcao de projecao
    lens = mapper.project(
        nPointCloud, # PointCloud normalizado
        projection=[1] # Projecao
    )

    # Criacao do complexo simplicial
    graph = mapper.map(
        lens,
        cover = km.Cover(n_cubes=1, perc_overlap=0.8),
        clusterer = DBSCAN()
    )

    # Criar a visualizacao
    mapper.visualize(
        graph,
        title="Teste de labels",
        path_html="Output.html",

        # !! Aqui voce coloca as suas labels !!
        custom_tooltips = Labels
    )

#-------------------------------------------------------#

if __name__ == '__main__':
    main()