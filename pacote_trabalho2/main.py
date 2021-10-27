# Import de bibliotecas utilizadas
import numpy as np
import sys
import timeit
import cv2
from numpy.lib.type_check import imag

INPUT_IMAGE1 = '/Users/thomszk/Desktop/PDI/PDI_2021.2/pacote_trabalho2/a01 - Original.bmp'
INPUT_IMAGE2 = '/Users/thomszk/Desktop/PDI/PDI_2021.2/pacote_trabalho2/b01 - Original.bmp'

# Tamanho do kernel NxN
KERNEL_WIDTH = 5
KERNEL_HEIGHT = 5



# Algoritmo filtro da media ingenuo ** Para imagens coloridas, processar cada canal RGB independentemente. !!EH BGR!!
def blurIngenuo(image, kernel_h, kernel_w):

    img = image.copy()
    media = kernel_h * kernel_w
    kernel_h = kernel_h // 2
    kernel_w = kernel_w // 2

    # Separando os canais de cores
    b, g, r = cv2.split(img)

    # test
    # print(img.shape)
    # print(len(image))
    # print(len(image[0]))

    # Fazendo o blur
    for row in range(len(img)):
        for col in range(len(img[0])):
            somaB = 0
            somaG = 0
            somaR = 0
            
            for y in range(row - kernel_h, row + kernel_h + 1):
                for x in range(col - kernel_w, col + kernel_w + 1):
                    if(y >= 0 and y < len(img) and x >= 0 and x < len(img[0])):
                        somaB += b[y][x]
                        somaG += g[y][x]
                        somaR += r[y][x]        

                img[row][col][0] = (somaB / media)
                img[row][col][1] = (somaG / media)
                img[row][col][2] = (somaR / media)
    return img

def blurSeparado(image, kernel_h, kernel_w):
    
    img = image.copy()
    height = kernel_h
    width = kernel_w
    kernel_h = kernel_h // 2
    kernel_w = kernel_w // 2

    # Separando os canais de cores
    b, g, r = cv2.split(img)

    # Fazendo blur HEIGTH
    for row in range(len(img)):
        for col in range(len(img[0])):
            somaB = 0
            somaG = 0
            somaR = 0
            
            for y in range(row - kernel_h, row + kernel_h + 1):
                if(y >= 0 and y < len(img)):
                    somaB += b[y][col]
                    somaG += g[y][col]
                    somaR += r[y][col]
                
            img[row][col][0] = (somaB / height)
            img[row][col][1] = (somaG / height)
            img[row][col][2] = (somaR / height)
    
    b, g, r = cv2.split(img)

    # Fazendo blur WIDTH
    for row in range(len(img)):
        for col in range(len(img[0])):
            somaB = 0
            somaG = 0
            somaR = 0
            
            for x in range(col - kernel_w, col + kernel_w + 1):
                if(x >= 0 and x < len(img[0])):
                    somaB += b[row][x]
                    somaG += g[row][x]
                    somaR += r[row][x]
                
            img[row][col][0] = (somaB / width)
            img[row][col][1] = (somaG / width)
            img[row][col][2] = (somaR / width)

    return img

def imagemIntegral(image):
    
    img = image.copy()

    # Separando canais de cores
    b, g, r = cv2.split(image)

    for row in range(len(image)):
        img[row][0][0] = b[row][0]
        img[row][0][1] = g[row][0]
        img[row][0][2] = r[row][0]
        
        for col in range(1, len(image[0])):
            img[row][col][0] = b[row][col] + img[row][col-1][0]
            img[row][col][1] = g[row][col] + img[row][col-1][1]
            img[row][col][2] = r[row][col] + img[row][col-1][2]

    for row in range(1, len(image)):
        for col in range(len(image[0])):
            img[row][col][0] = img[row][col][0] + img[row-1][col][0]
            img[row][col][1] = img[row][col][1] + img[row-1][col][1]
            img[row][col][2] = img[row][col][2] + img[row-1][col][2]


    return img

def filtroIntegral(image, kernel_h, kernel_w):          # Problema ans bordas!!!!! corrigir se der tempo.

    img = imagemIntegral(image)
    b, g, r = cv2.split(img)
    final = img.copy()
    media = kernel_w * kernel_h
    kernel_w = kernel_w // 2
    kernel_h = kernel_h // 2

    # if((row - kernel_h) < 0):
    #             sub_ht = -(row - kernel_h)
    #         if((row + kernel_h) > len(image)):
    #             sub_hb = -(len(image) - row)
    #         if((col - kernel_w) < 0):
    #             sub_wl = -(col - kernel_w)
    #         if((col + kernel_w) > len(image[0])):
    #             sub_wr = -(len(image[0]) - col)
    #         final[row][col][0] = b[row + kernel_h - sub_hb][col + kernel_w - sub_wr] - b[row + kernel_h - sub_hb][col - kernel_w + sub_wl] - b[row - kernel_h + sub_ht][col + kernel_w - sub_wr] + b[row - kernel_h + sub_ht][col - kernel_w + sub_wl]
    #         final[row][col][1] = g[row + kernel_h - sub_hb][col + kernel_w - sub_wr] - g[row + kernel_h - sub_hb][col - kernel_w + sub_wl] - g[row - kernel_h + sub_ht][col + kernel_w - sub_wr] + g[row - kernel_h + sub_ht][col - kernel_w + sub_wl]
    #         final[row][col][2] = r[row + kernel_h - sub_hb][col + kernel_w - sub_wr] - r[row + kernel_h - sub_hb][col - kernel_w + sub_wl] - r[row - kernel_h + sub_ht][col + kernel_w - sub_wr] + r[row - kernel_h + sub_ht][col - kernel_w + sub_wl]
    
    for row in range(kernel_h, len(image) - kernel_h):
        for col in range(kernel_w, len(image[0]) - kernel_w):
            final[row][col][0] = b[row + kernel_h][col + kernel_w] - b[row + kernel_h][col - kernel_w] - b[row - kernel_h][col + kernel_w] + b[row - kernel_h][col - kernel_w]
            final[row][col][1] = g[row + kernel_h][col + kernel_w] - g[row + kernel_h][col - kernel_w] - g[row - kernel_h][col + kernel_w] + g[row - kernel_h][col - kernel_w]
            final[row][col][2] = r[row + kernel_h][col + kernel_w] - r[row + kernel_h][col - kernel_w] - r[row - kernel_h][col + kernel_w] + r[row - kernel_h][col - kernel_w]
    
    final = final / media 
    return final

    

def main():
    # Abre as 2 imagens de exemplo disponibilizadas
    image1 = cv2.imread(INPUT_IMAGE1)
    image2 = cv2.imread(INPUT_IMAGE2)
    
    # Testa erro ao abrir uma imagem
    if image1 is None or image2 is None:
        print('Erro abrindo a alguma das imagens.\n')
        sys.exit()
    
    # parametriza as imagens
    image1 = image1.astype(np.float32) / 255
    image2 = image2.astype(np.float32) / 255

    # Aplica Filtro Media Ingenuo nas imagens 
    start_time = timeit.default_timer ()
    image1_ingenuo = blurIngenuo(image1, KERNEL_HEIGHT, KERNEL_WIDTH)
    print ('Tempo imagem 1 INGENUO: %f' % (timeit.default_timer () - start_time))

    start_time = timeit.default_timer ()
    image2_ingenuo = blurIngenuo(image2, KERNEL_HEIGHT, KERNEL_WIDTH)
    print ('Tempo imagem 2 INGENUO: %f' % (timeit.default_timer () - start_time))
    
    # Aplica Filtro Media Separavel nas imagens 
    start_time = timeit.default_timer ()
    image1_separavel = blurSeparado(image1, KERNEL_HEIGHT, KERNEL_WIDTH)
    print ('Tempo imagem 1 SEPARAVEL: %f' % (timeit.default_timer () - start_time))

    start_time = timeit.default_timer ()
    image2_separavel = blurSeparado(image2, KERNEL_HEIGHT, KERNEL_WIDTH)
    print ('Tempo imagem 2 SEPARAVEL: %f' % (timeit.default_timer () - start_time))

    # start_time = timeit.default_timer ()
    # image1_integral = imagemIntegral(image1)
    # print ('Tempo imagem 1 INTEGRAL: %f' % (timeit.default_timer () - start_time))

    # start_time = timeit.default_timer ()
    # image2_integral = imagemIntegral(image2)
    # print ('Tempo imagem 2 INTEGRAL: %f' % (timeit.default_timer () - start_time))

    # Faz a filtro integral da imagem
    start_time = timeit.default_timer ()
    image1_integral = filtroIntegral(image1, KERNEL_HEIGHT, KERNEL_WIDTH)
    print ('Tempo imagem 1 INTEGRAL: %f' % (timeit.default_timer () - start_time))
    
    start_time = timeit.default_timer ()
    image2_integral = filtroIntegral(image2, KERNEL_HEIGHT, KERNEL_WIDTH)
    print ('Tempo imagem 2 INTEGRAL: %f' % (timeit.default_timer () - start_time))

    cv2.imshow('01 - Original', image1)
    cv2.imshow('02 - Original', image2)
    cv2.imshow('01 - Ingenuo', image1_ingenuo)
    cv2.imshow('02 - Ingenuo', image2_ingenuo)
    cv2.imshow('01 - Separavel', image1_separavel)
    cv2.imshow('02 - Separavel', image2_separavel)
    cv2.imshow('01 - Integral', image1_integral)
    cv2.imshow('02 - Integral', image2_integral)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
