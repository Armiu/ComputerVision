import numpy as np
from scipy.fft import fft2, ifftshift, ifft2, fftshift


def gaussian_kernel(size, sigma):
    """
        Построение ядра фильтра Гаусса.

        @param  size  int    размер фильтра
        @param  sigma float  параметр размытия
        @return numpy array  фильтр Гаусса размером size x size
    """
    center = size // 2
    norm = np.linspace(- center, center, size)
    ox, oy = np.meshgrid(norm, norm)
    
    # Вычисление ядра Гаусса
    kernel = np.exp(-(ox**2 + oy**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    
    return kernel

def fourier_transform(h, shape):
    """
    Вычисляет Фурье-образ искажающей функции.

    Args:
        h: Искажающая функция (матрица).
        image_size: Размер изображения.

    Returns:
        Фурье-образ искажающей функции.
    """
   # Дополняем матрицу h нулями до нужного размера
    
    h_padded = np.pad(h, (
        (((shape[0] - h.shape[:2][0]) + 1) // 2, (shape[0] - h.shape[:2][0]) // 2),
        (((shape[1] - h.shape[:2][1]) + 1) // 2, (shape[1] - h.shape[:2][1]) // 2)
    ))
    h_padded = ifftshift(h_padded)
    # Вычисляем Фурье-образ
    H = fft2(h_padded)
    
    return H

def inverse_kernel(H, threshold=1e-10):
    """
        Получение H_inv

        @param  H            numpy array    Фурье-образ искажающей функции h
        @param  threshold    float          порог отсечения для избежания деления на 0
        @return numpy array  H_inv
    """
    # Создаем копию H, чтобы избежать изменения оригинального массива
    H_inv = np.copy(H)
    
    # Определяем индексы элементов, которые меньше или равны порогу
    idx1 = np.abs(H_inv) <= threshold
    idx2 = np.abs(H_inv) > threshold
    
    # Устанавливаем элементы, которые меньше или равны порогу, в 0
    H_inv[idx1] = 0
    # Инвертируем элементы, которые больше порога
    H_inv[idx2] = 1 / H_inv[idx2]
    
    return H_inv

def inverse_filtering(blurred_img, h, threshold=1e-10):
    """
        Метод инверсной фильтрации

        @param  blurred_img    numpy array  искаженное изображение
        @param  h              numpy array  искажающая функция
        @param  threshold      float        параметр получения H_inv
        @return numpy array                 восстановленное изображение
    """
    image_size = blurred_img.shape
    G = fft2(blurred_img)
    H = fourier_transform(h, image_size)
    # Получаем инверсный Фурье-образ искажающей функции
    F = G * inverse_kernel(H, threshold)
    f = ifft2(F)
    restored_img = np.abs(f)
    
    return restored_img

def wiener_filtering(blurred_img, h, K = 0.00009):
    """
    Метод оптимальной фильтрации по Винеру

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  K              float        константа для аппроксимации отношения спектров шума и неискаженного изображения
    @return numpy array                 восстановленное изображение
    """
    image_size = blurred_img.shape
    G = fft2(blurred_img)
    H = fourier_transform(h, image_size)
    
    # Получаем оценку  ̃F, используя формулу (7)
    H_conj = np.conj(H)
    # H_abs_squared = np.abs(H) ** 2
    F = (H_conj * G) / (H_conj * H + K)
  
    f = ifft2(F)
    restored_img = np.abs(f)
    
    return restored_img



def compute_psnr(img1, img2):
    """
        PSNR metric

        @param  img1    numpy array   оригинальное изображение
        @param  img2    numpy array   искаженное изображение
        @return float   PSNR(img1, img2)
    """
    max_pixel_value = 255
    # Вычисляем MSE (Mean Squared Error)
    mse = np.mean((img1 - img2) ** 2)
    
    if mse == 0:
        # Если MSE равно нулю, изображения идентичны
        return float('inf')
    
    # Вычисляем PSNR
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    
    return psnr
