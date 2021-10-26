# Import de bibliotecas utilizadas
import numpy as np
import sys
import timeit
import cv2

INPUT_IMAGE1 = '/Users/thomszk/Desktop/PDI/PDI_2021.2/pacote_trabalho2/a01 - Original.bmp'
INPUT_IMAGE2 = '/Users/thomszk/Desktop/PDI/PDI_2021.2/pacote_trabalho2/b01 - Original.bmp'

# Tamanho do kernel NxN
KERNEL_WIDTH = 3
KERNEL_HEIGHT = 3



# Algoritmo filtro da media ingenuo ** Para imagens coloridas, processar cada canal RGB independentemente. !!EH BGR!!
def blurIngenuo(image, kernel_h, kernel_w):

    img = image.copy()
    kernel_h = (kernel_h // 2)
    kernel_w = (kernel_w // 2)

    # Separando os canais de cores
    b, g, r = cv2.split(img)

    # test
    # print(img.shape)
    # print(len(image))
    # print(len(image[0]))

    # Fazendo o blur
    for row in range(len(image)):
        for col in range(len(image[0])):
            somaB = 0
            somaG = 0
            somaR = 0
            media = 0
            for y in range(row - kernel_h, row + kernel_h + 1):
                for x in range(col - kernel_w, col + kernel_w + 1):
                    if(y >= 0 and y < len(image) and x >= 0 and x < len(image[0])):
                        somaB += b[y][x]
                        somaG += g[y][x]
                        somaR += r[y][x]
                # if media == 0:
                    # media = 1
                    media += 1
                img[row][col][0] = (somaB / media)
                img[row][col][1] = (somaG / media)
                img[row][col][2] = (somaR / media)
    return img

def blurSeparado(image, box_height, box_width):
    ###PAREI AQUI
    print('ok')
    return None

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

    # Aplica Blur Ingenuo nas imagens 
    image1_out = blurIngenuo(image1, KERNEL_HEIGHT, KERNEL_WIDTH)
    image2_out = blurIngenuo(image2, KERNEL_HEIGHT, KERNEL_WIDTH)
    
    cv2.imshow('01 - Original', image1)
    cv2.imshow('02 - Original', image2)
    cv2.imshow('01 - Ingenuo', image1_out)
    cv2.imshow('02 - Ingenuo', image2_out)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
