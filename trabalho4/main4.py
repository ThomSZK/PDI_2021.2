from typing import final 
import numpy as np 
import sys 
import timeit 
import cv2
from numpy.lib.type_check import imag

INPUT_IMAGE1 = '/Users/thomszk/Desktop/PDI/PDI_2021.2/trabalho4/60.bmp'
INPUT_IMAGE2 = '/Users/thomszk/Desktop/PDI/PDI_2021.2/trabalho4/82.bmp'
INPUT_IMAGE3 = '/Users/thomszk/Desktop/PDI/PDI_2021.2/trabalho4/114.bmp'
INPUT_IMAGE4 = '/Users/thomszk/Desktop/PDI/PDI_2021.2/trabalho4/150.bmp'
INPUT_IMAGE5 = '/Users/thomszk/Desktop/PDI/PDI_2021.2/trabalho4/205.bmp'

KERNEL_SIZE = 45
THRESHOLD = 130
ALTURA_MIN = 3
LARGURA_MIN = 3
N_PIXELS_MIN = 10

# Limiarizacao Otsu apos aplicar o blur gaussiano
def otsu(img):

    # blur = cv2.GaussianBlur(img, (KERNEL_SIZE, KERNEL_SIZE), 0)
    thresh, output_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # output_otsu = output_otsu.astype(np.int8)
    return output_otsu

# Funcao de limiarizacao adaptativa (aplica box blur, faz diferenca com a original e limiariza)
def thresholding(img):

    original = img.copy()
    blur = img.copy()
    # img_blur = np.zeros(blur.shape, dtype = np.uint8)
    # blur = np.where(blur < 40, 1, blur)
    # for i in range (2):
    #     blur = cv2.GaussianBlur(blur,(2**(i+5) + 1, 2**(i+5) + 1), 0)
    #     img_blur += blur
    # blur = img_blur 
    blur = cv2.GaussianBlur(blur, (KERNEL_SIZE, KERNEL_SIZE), 0)

    # Faz a diferenca entre imagem original e blur e limiariza
    # cv2.imshow('TEST', blur)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    blur = blur - original

    # np.where(blur < 0, 0, blur)
    # cv2.imshow('TEST', blur)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    kernel = np.ones((2,2), np.uint8)
    thresh = cv2.erode(thresh, kernel)
    thresh = cv2.erode(thresh, kernel)
    thresh = cv2.erode(thresh, kernel)
    # thresh = cv2.erode(thresh, kernel)
    # thresh = cv2.erode(thresh, kernel)
    # thresh = cv2.erode(thresh, kernel)
    # thresh = cv2.erode(thresh, kernel)
    
    return thresh

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

def rotula (img, largura_min, altura_min, n_pixels_min):
    label = 1

    #separando background e foreground
    img = np.where(img == 1.0, -1, 0.0) #manter -1 ou -1.0 ??

    #varredura procurando pixel unlabeled
    for row in range(len(img) - 3):
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


def main():

    # Abre as imagens de exemplo disponibilizadas em escala de cinza
    image1 = cv2.imread(INPUT_IMAGE1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(INPUT_IMAGE2, cv2.IMREAD_GRAYSCALE)
    image3 = cv2.imread(INPUT_IMAGE3, cv2.IMREAD_GRAYSCALE)
    image4 = cv2.imread(INPUT_IMAGE4, cv2.IMREAD_GRAYSCALE)
    image5 = cv2.imread(INPUT_IMAGE5, cv2.IMREAD_GRAYSCALE)

    # Testa erro ao abrir uma imagem
    if image1 is None or image2 is None or image3 is None or image4 is None or image5 is None:
        print('!!!Erro abrindo a alguma das imagens!!!\n')
        sys.exit()

    # Parametriza as imagens
    # image1 = image1.astype(np.float32) 
    # image2 = image2.astype(np.float32) 
    # image3 = image3.astype(np.float32) 
    # image4 = image4.astype(np.float32) 
    # image5 = image5.astype(np.float32) 
    
    # Aplica a limiarizacao de Otsu
    start_time = timeit.default_timer()
    image1_otsu = otsu(image1)
    image2_otsu = otsu(image2)
    image3_otsu = otsu(image3)
    image4_otsu = otsu(image4)
    image5_otsu = otsu(image5)
    print ('Tempo limiarizacao otsu das imagens: %f' % (timeit.default_timer () - start_time))

    # Aplica a limiarizacao adaptativa usando box blur e limiarizacao binaria 
    start_time = timeit.default_timer()
    image1_adapt = thresholding(image1)
    image2_adapt = thresholding(image2)
    image3_adapt = thresholding(image3)
    image4_adapt = thresholding(image4)
    image5_adapt = thresholding(image5)
    print ('Tempo limiarizacao adaptativa das imagens: %f' % (timeit.default_timer () - start_time))

    # Parametriza as imagens
    image1_adapt = image1_adapt.astype(np.float32) / 255
    image2_adapt = image2_adapt.astype(np.float32) / 255
    image3_adapt = image3_adapt.astype(np.float32) / 255
    image4_adapt = image4_adapt.astype(np.float32) / 255
    image5_adapt = image5_adapt.astype(np.float32) / 255

    # start_time = timeit.default_timer ()
    # componentes = rotula (image1_adapt, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)
    # n_componentes = len (componentes)
    # print ('Tempo 60: %f' % (timeit.default_timer () - start_time))
    # print ('%d componentes detectados.' % n_componentes)

    start_time = timeit.default_timer ()
    componentes = rotula (image2_adapt, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)
    n_componentes = len (componentes)
    print ('Tempo 82: %f' % (timeit.default_timer () - start_time))
    print ('%d componentes detectados.' % n_componentes)

    start_time = timeit.default_timer ()
    componentes = rotula (image3_adapt, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)
    n_componentes = len (componentes)
    print ('Tempo 114: %f' % (timeit.default_timer () - start_time))
    print ('%d componentes detectados.' % n_componentes)

    start_time = timeit.default_timer ()
    componentes = rotula (image4_adapt, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)
    n_componentes = len (componentes)
    print ('Tempo 150: %f' % (timeit.default_timer () - start_time))
    print ('%d componentes detectados.' % n_componentes)

    start_time = timeit.default_timer ()
    componentes = rotula (image5_adapt, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)
    n_componentes = len (componentes)
    print ('Tempo 205: %f' % (timeit.default_timer () - start_time))
    print ('%d componentes detectados.' % n_componentes)

    cv2.imshow('60 - Original', image1)
    cv2.imshow('82 - Original', image2)
    cv2.imshow('114 - Original', image3)
    cv2.imshow('150 - Original', image4)
    cv2.imshow('205 - Original', image5)
    cv2.imshow('60 - Otsu', image1_otsu)
    cv2.imshow('82 - Otsu', image2_otsu)
    cv2.imshow('114 - Otsu', image3_otsu)
    cv2.imshow('150 - Otsu', image4_otsu)
    cv2.imshow('205 - Otsu', image5_otsu)
    cv2.imshow('60 - Adapt', image1_adapt)
    cv2.imshow('82 - Adapt', image2_adapt)
    cv2.imshow('114 - Adapt', image3_adapt)
    cv2.imshow('150 - Adapt', image4_adapt)
    cv2.imshow('205 - Adapt', image5_adapt)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()