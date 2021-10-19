#===============================================================================
# Exemplo: segmentação de uma imagem em escala de cinza.
#-------------------------------------------------------------------------------
# Autor: Bogdan T. Nassu
# Universidade Tecnológica Federal do Paraná
#===============================================================================

import sys
import timeit
import numpy as np
import cv2

#===============================================================================

INPUT_IMAGE =  'arroz.bmp'

# TODO: ajuste estes parâmetros!
NEGATIVO = False
THRESHOLD = 0.82
ALTURA_MIN = 3
LARGURA_MIN = 3
N_PIXELS_MIN = 5

#===============================================================================

def binariza (img, threshold):
    ''' Binarização simples por limiarização.

Parâmetros: img: imagem de entrada. Se tiver mais que 1 canal, binariza cada
              canal independentemente.
            threshold: limiar.
            
Valor de retorno: versão binarizada da img_in.'''

    # TODO: escreva o código desta função.
    # Dica/desafio: usando a função np.where, dá para fazer a binarização muito
    # rapidamente, e com apenas uma linha de código!

    imgb = np.where(img>threshold, 1.0, 0.0)
    
    return imgb

#-------------------------------------------------------------------------------

def rotula (img, largura_min, altura_min, n_pixels_min):
    '''Rotulagem usando flood fill. Marca os objetos da imagem com os valores
[0.1,0.2,etc].

Parâmetros: img: imagem de entrada E saída.
            largura_min: descarta componentes com largura menor que esta.
            altura_min: descarta componentes com altura menor que esta.
            n_pixels_min: descarta componentes com menos pixels que isso.

Valor de retorno: uma lista, onde cada item é um vetor associativo (dictionary)
com os seguintes campos:

'label': rótulo do componente.
'n_pixels': número de pixels do componente.
'T', 'L', 'B', 'R': coordenadas do retângulo envolvente de um componente conexo,
respectivamente: topo, esquerda, baixo e direita.'''

    # TODO: escreva esta função.
    # Use a abordagem com flood fill recursivo.

    label = 1

    #separando background e foreground
    img = np.where(img == 1.0, -1, 0.0) #manter -1 ou -1.0 ??

    #varredura procurando pixel unlabeled
    for row in range(len(img)):
        for col in range(len(img[0])):
            if img[row][col] == -1:
                inunda(label, img, row, col)
                label += 1
    
    #cria lista/dicionario com cada blob filled (arroz)
    graos = list()
    for i in range(label):
        graos.append({'label': i, 'n_pixels': 0, 'T': 300000, 'L': -1, 'B': -1, 'R': 3000000})
    
    #contando o numero de pixels com o label encontrado e TLBR
    for row in range(len(img)):
        for col in range(len(img[0])):
            if img[row][col] >= 1:
                tag = int(img[row][col])
                graos[tag]['n_pixels'] += 1 
                if graos[tag]['T'] > row:
                    graos[tag]['T'] = row
                if graos[tag]['L'] < col:
                    graos[tag]['L'] = col
                if graos[tag]['B'] < row:
                    graos[tag]['B'] = row
                if graos[tag]['R'] > col:
                    graos[tag]['R'] = col
    
    #filtrando a lista de graos baseado nos parametros 
    graos_filtered = list()
    
    for i in range(label):
        if ((graos[i]['n_pixels'] >= N_PIXELS_MIN) and ((graos[i]['B'] - graos[i]['T']) >= ALTURA_MIN) and (( graos[i]['L'] - graos[i]['R']) >= LARGURA_MIN)):
            graos_filtered.append(graos[i])
    # print(graos_filtered)
    return graos_filtered


#vizinhanca - 4 - recursivo 
def inunda(label, f, x0, y0):
    
    f[x0][y0] = label
    if f[x0 + 1][y0] == -1 and x0 + 1 <= len(f):
        inunda(label, f, x0 + 1, y0)
    if f[x0 - 1][y0] == -1 and x0 + 1 >= 0:
        inunda(label, f, x0 - 1, y0)
    if f[x0][y0 + 1] == -1 and y0 + 1 <= len(f[0]):
        inunda(label, f, x0, y0 + 1)
    if f[x0][y0 - 1] == -1 and x0 + 1 >= 0:
        inunda(label, f, x0, y0 - 1)

#===============================================================================

def main ():

    # Abre a imagem em escala de cinza.
    img = cv2.imread(INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()
    #test
    cv2.imshow('image', img)
    # É uma boa prática manter o shape com 3 valores, independente da imagem ser
    # colorida ou não. Também já convertemos para float32.
    img = img.reshape ((img.shape [0], img.shape [1], 1))
    img = img.astype (np.float32) / 255

    # Mantém uma cópia colorida para desenhar a saída.
    img_out = cv2.cvtColor (img, cv2.COLOR_GRAY2BGR)

    # Segmenta a imagem.
    if NEGATIVO:
        img = 1 - img
    img = binariza (img, THRESHOLD)
    cv2.imshow ('01 - binarizada', img)
    cv2.imwrite ('01 - binarizada.png', img*255)

    start_time = timeit.default_timer ()
    componentes = rotula (img, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)
    n_componentes = len (componentes)
    print ('Tempo: %f' % (timeit.default_timer () - start_time))
    print ('%d componentes detectados.' % n_componentes)

    # Mostra os objetos encontrados.
    for c in componentes:
        cv2.rectangle (img_out, (c ['L'], c ['T']), (c ['R'], c ['B']), (0,0,1))

    cv2.imshow ('02 - out', img_out)
    cv2.imwrite ('02 - out.png', img_out*255)
    cv2.waitKey ()
    cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()

#===============================================================================
