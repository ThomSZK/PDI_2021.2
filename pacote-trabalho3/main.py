from typing import final
import numpy as np
import sys
import timeit
import cv2
from numpy.lib.type_check import imag


INPUT_IMAGE1 = '/Users/thomszk/Desktop/PDI/PDI_2021.2/pacote-trabalho3/GT2.BMP'
INPUT_IMAGE2 = '/Users/thomszk/Desktop/PDI/PDI_2021.2/pacote-trabalho3/Wind Waker GC.bmp'

KERNEL_WIDTH = 19
KERNEL_HEIGHT = 19
THRESHOLD = 0.58

# Faz a mascara, setando tudo abaixo do threshold pra 0 e mantendo os outros valores.
def makeMask(img):

    original = img.copy()
    mask = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    # mask = cv2.inRange(img, (THRESHOLD), (1))
    # maskb = np.where(mask > THRESHOLD, 1, 0)
    ret, thresh1 = cv2.threshold(mask, THRESHOLD, 1, cv2.THRESH_BINARY)
    thresh1 = thresh1.astype(np.int8)
    # ret, thresh2 = cv2.threshold(mask, THRESHOLD, 1, cv2.THRESH_BINARY_INV)
    # ret, thresh3 = cv2.threshold(mask, THRESHOLD, 1, cv2.THRESH_TRUNC)
    # ret, thresh4 = cv2.threshold(mask, THRESHOLD, 1, cv2.THRESH_TOZERO)
    # ret, thresh5 = cv2.threshold(mask, THRESHOLD, 1, cv2.THRESH_TOZERO_INV)
    final = cv2.bitwise_and(original, original, mask = thresh1)

    return final

# Funcao de Bloom usando o box blur
def box(img, mask):
    
    original = img.copy()
    image = img.copy()
    blur = mask.copy()

    # Executa o blur 3x com kernel de 19x19
    for i in range(3):
        blur += cv2.blur(blur, (KERNEL_HEIGHT, KERNEL_WIDTH))
    
    # Ajustar os parametros da soma para aumentar ou diminuir bloom 
    image = image + blur
    # image *= 1/image.max()
    # norm_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    final = (0.97 * original) + (0.03 * image) 

    return final

# Funcao de Bloom usando o Gaussian blur
def gaussiano(img, mask):

    original = img.copy()
    image = img.copy()
    maskd = mask.copy()

    # Tentei mudar os parametros mas o gaussiano parece ser menos 'soft' do que o box
    maskd = cv2.GaussianBlur(maskd, (13, 13), 10, 10, cv2.BORDER_DEFAULT)
    
    image = image + maskd
    # Ajustar os parametros da soma para aumentar ou diminuir bloom 
    final = (0.8 * original) + (0.2 * image) 

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

    # Faz a mask
    start_time = timeit.default_timer ()
    image1_mask = makeMask(image1)
    print ('Tempo imagem 1 Mask: %f' % (timeit.default_timer () - start_time))

    start_time = timeit.default_timer ()
    image2_mask = makeMask(image2)
    print ('Tempo imagem 2 Mask: %f' % (timeit.default_timer () - start_time))

    # Aplica Bloom com Box Blur 
    start_time = timeit.default_timer ()
    image1_box = box(image1 ,image1_mask)
    print ('Tempo imagem 1 Mask: %f' % (timeit.default_timer () - start_time))

    start_time = timeit.default_timer ()
    image2_box = box(image2, image2_mask)
    print ('Tempo imagem 2 Mask: %f' % (timeit.default_timer () - start_time))

    # Aplica Bloom com Gaussian Blur 
    start_time = timeit.default_timer ()
    image1_gauss = gaussiano(image1 ,image1_mask)
    print ('Tempo imagem 1 Mask: %f' % (timeit.default_timer () - start_time))

    start_time = timeit.default_timer ()
    image2_gauss = gaussiano(image2, image2_mask)
    print ('Tempo imagem 2 Mask: %f' % (timeit.default_timer () - start_time))

    cv2.imshow('01 - Original', image1)
    cv2.imshow('02 - Original', image2)
    cv2.imshow('01 - Mask', image1_mask)
    cv2.imshow('02 - Mask', image2_mask)
    cv2.imshow('01 - Box', image1_box)
    cv2.imshow('02 - Box', image2_box)
    cv2.imshow('01 - Gaussian', image1_gauss)
    cv2.imshow('02 - Gaussian', image2_gauss)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()