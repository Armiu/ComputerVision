import numpy as np
from scipy.ndimage import convolve

def get_bayer_masks(n_rows, n_cols):
    red_mask = np.zeros((n_rows, n_cols), dtype=bool)
    green_mask = np.zeros((n_rows, n_cols), dtype=bool)
    blue_mask = np.zeros((n_rows, n_cols), dtype=bool)

    red_mask[::2, 1::2] = True
    blue_mask[1::2, ::2] = True
    green_mask[::2, ::2] = True
    green_mask[1::2, 1::2] = True

    masks = np.stack((red_mask, green_mask, blue_mask), axis=-1)
    return masks

def get_colored_img(raw_img):
    n_rows, n_cols = raw_img.shape
    masks = get_bayer_masks(n_rows, n_cols)

    # Инициализируем трехканальное изображение
    colored_img = np.zeros((n_rows, n_cols, 3), dtype=raw_img.dtype)

    # Заполняем цветовые каналы согласно маскам
    for channel in range(3):
        colored_img[:, :, channel] = np.where(masks[:, :, channel], raw_img, 0)

    return colored_img

def get_raw_img(colored_img):
    n_rows, n_cols, _ = colored_img.shape
    masks = get_bayer_masks(n_rows, n_cols)

    # Инициализируем одноканальное изображение
    raw_img = np.zeros((n_rows, n_cols), dtype=colored_img.dtype)

    # Заполняем одноканальное изображение согласно маскам
    for channel in range(3):
        raw_img += np.where(masks[:, :, channel], colored_img[:, :, channel], 0)

    return raw_img

from scipy.ndimage import convolve

def bilinear_interpolation(raw_img):
    n_rows, n_cols = raw_img.shape
    rgb_img = np.zeros((n_rows, n_cols, 3), dtype=np.float64)
    
    # Создание масок для каждого канала
    red_mask = np.zeros_like(raw_img, dtype=bool)
    blue_mask = np.zeros_like(raw_img, dtype=bool)
    green_mask = np.zeros_like(raw_img, dtype=bool)
    
    red_mask[::2, 1::2] = True
    blue_mask[1::2, ::2] = True
    green_mask[::2, ::2] = True
    green_mask[1::2, 1::2] = True
    
    # Заполнение известных значений в rgb_img
    rgb_img[:, :, 0][red_mask] = raw_img[red_mask]
    rgb_img[:, :, 1][green_mask] = raw_img[green_mask]
    rgb_img[:, :, 2][blue_mask] = raw_img[blue_mask]
    
    # Определение ядра свертки
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.float64)
    
    def interpolate_channel(channel, mask):
        # Свертка для суммирования известных значений и подсчета количества известных значений
        known_values = convolve(channel * mask, kernel, mode='constant', cval=0.0)
        counts = convolve(mask.astype(np.float64), kernel, mode='constant', cval=0.0)
        
        # Вычисление среднего значения
        interpolated_values = known_values / np.maximum(counts, 1)
        channel[~mask] = interpolated_values[~mask]
    
    # Интерполяция для каждого канала
    interpolate_channel(rgb_img[:, :, 0], red_mask)
    interpolate_channel(rgb_img[:, :, 1], green_mask)
    interpolate_channel(rgb_img[:, :, 2], blue_mask)
    
    # Приведение значений к диапазону [0, 255] и преобразование в uint8
    rgb_img = np.clip(rgb_img, 0, 255).astype(np.uint8)
    
    return rgb_img



# def improved_interpolation(raw_image):
#     h = raw_image.shape[0]
#     w = raw_image.shape[1]
    
#     weights_0 = (1 / 8) * np.array([
#         [ 0,  0, 1/2,  0,  0],
#         [ 0, -1,   0, -1,  0],
#         [-1,  4,   5,  4, -1],
#         [ 0, -1,   0, -1,  0],
#         [ 0,  0, 1/2,  0,  0]
#     ])

#     weights_1 = (1 / 8) * np.array([
#         [  0,  0,  -1,  0,   0],
#         [  0, -1,   4, -1,   0],
#         [1/2,  0,   5,  0, 1/2],
#         [  0, -1,   4, -1,   0],
#         [  0,  0,  -1,  0,   0]
#     ])

#     weights_2 = (1 / 8) * np.array([
#         [   0,  0, -3/2,  0,    0],
#         [   0,  2,    0,  2,    0],
#         [-3/2,  0,    6,  0, -3/2],
#         [   0,  2,    0,  2,    0],
#         [   0,  0, -3/2,  0,    0]
#     ])

#     weights_3 = (1 / 8) * np.array([
#         [ 0,  0, -1,  0,  0],
#         [ 0,  0,  2,  0,  0],
#         [-1,  2,  4,  2, -1],
#         [ 0,  0,  2,  0,  0],
#         [ 0,  0, -1,  0,  0]
#     ])

#     image = np.zeros((h, w, 3), dtype=float)
#     masks = get_bayer_masks(h, w)
#     mask_r, mask_g1, mask_g2, mask_b = masks[..., 0], masks[..., 1], masks[..., 1], masks[..., 2]
   
#     # Применяем padding к изображению, чтобы корректно обрабатывать границы
#     padded_img = np.pad(raw_image, 2, 'constant', constant_values=0)
    
#     # Интерполяция зеленого канала в позициях красного и синего пикселей
#     green_interpolated = convolve(padded_img, weights_0, mode='constant', cval=0.0)
#     image[..., 1] = np.where(mask_g1 | mask_g2, raw_image, green_interpolated[2:-2, 2:-2])
    
#     # Интерполяция красного канала в позициях зеленых пикселей
#     red_interpolated = convolve(padded_img, weights_1, mode='constant', cval=0.0)
#     image[..., 0] = np.where(mask_r, raw_image, red_interpolated[2:-2, 2:-2])
    
#     # Интерполяция синего канала в позициях зеленых пикселей
#     blue_interpolated = convolve(padded_img, weights_2, mode='constant', cval=0.0)
#     image[..., 2] = np.where(mask_b, raw_image, blue_interpolated[2:-2, 2:-2])
    
#     # Интерполяция красного канала в позициях синих пикселей
#     red_at_blue_interpolated = convolve(padded_img, weights_3, mode='constant', cval=0.0)
#     image[..., 0] = np.where(mask_b, red_at_blue_interpolated[2:-2, 2:-2], image[..., 0])
    
#     # Интерполяция синего канала в позициях красных пикселей
#     blue_at_red_interpolated = convolve(padded_img, weights_3, mode='constant', cval=0.0)
#     image[..., 2] = np.where(mask_r, blue_at_red_interpolated[2:-2, 2:-2], image[..., 2])
    
#     return np.clip(image, 0, 255).astype(np.uint8)



def improved_interpolation(raw_image):
    h = raw_image.shape[0]
    w = raw_image.shape[1]
    

    weigths_0 = (1 / 8) * np.array([
        [ 0,  0, 1/2,  0,  0],
        [ 0, -1,   0, -1,  0],
        [-1,  4,   5,  4, -1],
        [ 0, -1,   0, -1,  0],
        [ 0,  0, 1/2,  0,  0]
    ])

    weigths_1 = (1 / 8) * np.array([
        [  0,  0,  -1,  0,   0],
        [  0, -1,   4, -1,   0],
        [1/2,  0,   5,  0, 1/2],
        [  0, -1,   4, -1,   0],
        [  0,  0,  -1,  0,   0]
    ])

    weigths_2 = (1 / 8) * np.array([
        [   0,  0, -3/2,  0,    0],
        [   0,  2,    0,  2,    0],
        [-3/2,  0,    6,  0, -3/2],
        [   0,  2,    0,  2,    0],
        [   0,  0, -3/2,  0,    0]
    ])

    weigths_3 = (1 / 8) * np.array([
        [ 0,  0, -1,  0,  0],
        [ 0,  0,  2,  0,  0],
        [-1,  2,  4,  2, -1],
        [ 0,  0,  2,  0,  0],
        [ 0,  0, -1,  0,  0]
    ])

    image = np.zeros((h, w, 3), dtype=float)
    masks = get_bayer_masks(h, w)
    mask_r, mask_g1, mask_g2, mask_b = masks[..., 0], masks[..., 1], masks[..., 1], masks[..., 2]
   
   # YOUR CODE
    g1m=np.pad(np.multiply(raw_image,mask_g1),pad_width=2,mode='constant',constant_values=0)
    g2m=np.pad(np.multiply(raw_image,mask_g2),pad_width=2,mode='constant',constant_values=0)
    rm=np.pad(np.multiply(raw_image,mask_r),pad_width=2,mode='constant',constant_values=0)
    bm=np.pad(np.multiply(raw_image,mask_b),pad_width=2,mode='constant',constant_values=0)
    k=np.pad(raw_image,pad_width=2,mode='constant',constant_values=0)
    
#может быть тут нужно было использовать свертки, но я не представляю как.
#Я не считаю, что тут вложенные циклы - плохое решение, т.к. я не перемножаю 
# нулевые элементы
    g=g1m+g2m

    for i in range(2,h+2,2):   #blue
        for j in range(2,w+2,2):
            g[i][j]=k[i][j]/2+(g[i+1][j]+g[i][j+1]+g[i-1][j]+g[i][j-1])/4-(k[i+2][j]+k[i][j+2]+k[i-2][j]+k[i][j-2])/8
    for i in range(3,h+2,2):  #red
        for j in range(3,w+2,2):
            g[i][j]=k[i][j]/2+(g[i+1][j]+g[i][j+1]+g[i-1][j]+g[i][j-1])/4-(k[i+2][j]+k[i][j+2]+k[i-2][j]+k[i][j-2])/8
            
    b=bm
    for i in range(2,h+2,2):   #rows
        for j in range(3,w+2,2):
            b[i][j]=k[i][j]*5/8+(b[i][j+1]+b[i][j-1])/2-(k[i+1][j-1]+k[i+1][j+1]+k[i-1][j-1]+k[i-1][j+1])/8-(k[i][j-2]+k[i][j+2])/8+(k[i+2][j]+k[i-2][j])/16 

    for i in range(3,h+2,2):   #coloms
        for j in range(2,w+2,2):
            b[i][j]=k[i][j]*5/8+(b[i+1][j]+b[i-1][j])/2-(k[i+1][j-1]+k[i+1][j+1]+k[i-1][j-1]+k[i-1][j+1])/8+(k[i][j-2]+k[i][j+2])/16-(k[i+2][j]+k[i-2][j])/8 

    for i in range(3,h+2,2):   #red
        for j in range(3,w+2,2):
            b[i][j]=k[i][j]*3/4+(b[i+1][j-1]+b[i+1][j+1]+b[i-1][j-1]+b[i-1][j+1])/4-(k[i][j-2]+k[i][j+2]+k[i+2][j]+k[i-2][j])*3/16 

    r=rm
    for i in range(3,h+2,2):   #rows
        for j in range(2,w+2,2):
            r[i][j]=k[i][j]*5/8+(r[i][j+1]+r[i][j-1])/2-(k[i+1][j-1]+k[i+1][j+1]+k[i-1][j-1]+k[i-1][j+1])/8-(k[i][j-2]+k[i][j+2])/8+(k[i+2][j]+k[i-2][j])/16 

    for i in range(2,h+2,2):   #coloms
        for j in range(3,w+2,2):
            r[i][j]=k[i][j]*5/8+(r[i+1][j]+r[i-1][j])/2-(k[i+1][j-1]+k[i+1][j+1]+k[i-1][j-1]+k[i-1][j+1])/8+(k[i][j-2]+k[i][j+2])/16-(k[i+2][j]+k[i-2][j])/8 

    for i in range(2,h+2,2):   #blue
        for j in range(2,w+2,2):
            r[i][j]=k[i][j]*3/4+(r[i+1][j-1]+r[i+1][j+1]+r[i-1][j-1]+r[i-1][j+1])/4-(k[i][j-2]+k[i][j+2]+k[i+2][j]+k[i-2][j])*3/16 
   
    image[:,:,0]=r[2:h+2,2:w+2]
    image[:,:,1]=g[2:h+2,2:w+2]
    image[:,:,2]=b[2:h+2,2:w+2]
    return np.clip(image, 0, 255).astype(np.uint8)


# def improved_interpolation(raw_img):
#     """
#     Реализация улучшенной линейной интерполяции по статье Malvar, He, Cutler.

#     Args:
#         raw_img: Массив NumPy, представляющий изображение Байера (одноканальное).

#     Returns:
#         Массив NumPy, представляющий интерполированное цветное изображение (трёхканальное).
#     """

#     raw_img = raw_img.astype(np.float64)  # Приводим к типу float64
#     height, width = raw_img.shape

#     # Создаем пустое изображение для интерполированных значений
#     interpolated_img = np.zeros((height, width, 3), dtype=np.float64)

#     # Получаем маски для каждого канала
#     r_mask, g_mask, b_mask = get_bayer_masks(height, width).transpose(2, 0, 1)

#     # Заполняем известные значения
#     interpolated_img[:, :, 0][r_mask] = raw_img[r_mask]
#     interpolated_img[:, :, 1][g_mask] = raw_img[g_mask]
#     interpolated_img[:, :, 2][b_mask] = raw_img[b_mask]

#     # Определяем фильтры для интерполяции (из статьи)
#     g_at_r_filter = (1 / 8) * np.array([
#         [ 0,  0, 1/2,  0,  0],
#         [ 0, -1,   0, -1,  0],
#         [-1,  4,   5,  4, -1],
#         [ 0, -1,   0, -1,  0],
#         [ 0,  0, 1/2,  0,  0]
#     ])

#     g_at_b_filter = (1 / 8) * np.array([
#         [  0,  0,  -1,  0,   0],
#         [  0, -1,   4, -1,   0],
#         [1/2,  0,   5,  0, 1/2],
#         [  0, -1,   4, -1,   0],
#         [  0,  0,  -1,  0,   0]
#     ])

#     r_at_b_filter = (1 / 8) * np.array([
#         [   0,  0, -3/2,  0,    0],
#         [   0,  2,    0,  2,    0],
#         [-3/2,  0,    6,  0, -3/2],
#         [   0,  2,    0,  2,    0],
#         [   0,  0, -3/2,  0,    0]
#     ])

#     b_at_r_filter = (1 / 8) * np.array([
#         [ 0,  0, -1,  0,  0],
#         [ 0,  0,  2,  0,  0],
#         [-1,  2,  4,  2, -1],
#         [ 0,  0,  2,  0,  0],
#         [ 0,  0, -1,  0,  0]
#     ])

#     # Применяем padding к изображению, чтобы корректно обрабатывать границы
#     padded_img = np.pad(raw_img, 2, 'reflect')

#     # Интерполяция зеленого канала в позициях красного и синего пикселей
#     interpolated_img[..., 1] = np.where(r_mask | b_mask, 
#                                         convolve(padded_img, g_at_r_filter, mode='reflect')[2:-2, 2:-2], 
#                                         interpolated_img[..., 1])

#     # Интерполяция красного канала в позициях зеленых пикселей
#     interpolated_img[..., 0] = np.where(g_mask, 
#                                         convolve(padded_img, r_at_b_filter, mode='reflect')[2:-2, 2:-2], 
#                                         interpolated_img[..., 0])

#     # Интерполяция синего канала в позициях зеленых пикселей
#     interpolated_img[..., 2] = np.where(g_mask, 
#                                         convolve(padded_img, b_at_r_filter, mode='reflect')[2:-2, 2:-2], 
#                                         interpolated_img[..., 2])

#     # Интерполяция красного канала в позициях синих пикселей
#     interpolated_img[..., 0] = np.where(b_mask, 
#                                         convolve(padded_img, r_at_b_filter, mode='reflect')[2:-2, 2:-2], 
#                                         interpolated_img[..., 0])

#     # Интерполяция синего канала в позициях красных пикселей
#     interpolated_img[..., 2] = np.where(r_mask, 
#                                         convolve(padded_img, b_at_r_filter, mode='reflect')[2:-2, 2:-2], 
#                                         interpolated_img[..., 2])

#     interpolated_img = np.clip(interpolated_img, 0, 255).astype(np.uint8)

#     return interpolated_img
# def improved_interpolation(raw_img):
#     """
#     Реализация улучшенной линейной интерполяции по статье Malvar, He, Cutler.

#     Args:
#         raw_img: Массив NumPy, представляющий изображение Байера (одноканальное).

#     Returns:
#         Массив NumPy, представляющий интерполированное цветное изображение (трёхканальное).
#     """

#     raw_img = raw_img.astype(np.float64)  # Приводим к типу float64
#     height, width = raw_img.shape

#     # Создаем пустое изображение для интерполированных значений
#     interpolated_img = np.zeros((height, width, 3), dtype=np.float64)

#     # Получаем маски для каждого канала
#     r_mask, g_mask, b_mask = get_bayer_masks(height, width).transpose(2, 0, 1)

#     # Заполняем известные значения
#     interpolated_img[:, :, 0][r_mask] = raw_img[r_mask]
#     interpolated_img[:, :, 1][g_mask] = raw_img[g_mask]
#     interpolated_img[:, :, 2][b_mask] = raw_img[b_mask]

#     # Определяем фильтры для интерполяции (из статьи)
#     g_at_r_filter = np.array([
#         [0,  0, -1,  0,  0],
#         [0,  0,  2,  0,  0],
#         [-1,  2,  4,  2, -1],
#         [0,  0,  2,  0,  0],
#         [0,  0, -1,  0,  0]
#     ]) / 8
#     r_at_g_in_rb_filter = np.array([
#         [-1, 4, -1],
#         [4, 5, 4],
#         [-1, 4, -1]
#     ]) / 8
#     b_at_g_in_rb_filter = r_at_g_in_rb_filter
#     r_at_b_filter = np.array([
#         [0,  0, -3/2,  0,  0],
#         [0,  2,  0,  2,  0],
#         [-3/2,  0,  6,  0, -3/2],
#         [0,  2,  0,  2,  0],
#         [0,  0, -3/2,  0,  0]
#     ]) / 8
#     b_at_r_filter = r_at_b_filter

#     # Применяем padding к изображению, чтобы корректно обрабатывать границы
#     padded_img = np.pad(raw_img, 2, 'reflect')

#     # Проходим по всем пикселям, кроме границы
#     for i in range(2, height + 2):
#         for j in range(2, width + 2):
#             # Интерполируем зеленый в позициях красного
#             if r_mask[i - 2, j - 2]:
#                 interpolated_img[i - 2, j - 2, 1] = np.sum(
#                     padded_img[i - 2:i + 3, j - 2:j + 3] * g_at_r_filter
#                 )
#             # Интерполируем зеленый в позициях синего
#             if b_mask[i - 2, j - 2]:
#                 interpolated_img[i - 2, j - 2, 1] = np.sum(
#                     padded_img[i - 2:i + 3, j - 2:j + 3] * g_at_r_filter
#                 )
#             # Интерполируем красный в позициях зеленого, где в той же строке синий
#             if g_mask[i - 2, j - 2] and b_mask[i - 2, j - 1]:
#                 interpolated_img[i - 2, j - 2, 0] = np.sum(
#                     padded_img[i - 3:i + 2, j - 3:j + 2] * r_at_g_in_rb_filter
#                 )
#             # Интерполируем красный в позициях зеленого, где в той же строке красный
#             if g_mask[i - 2, j - 2] and r_mask[i - 2, j - 3]:
#                 interpolated_img[i - 2, j - 2, 0] = np.sum(
#                     padded_img[i - 3:i + 2, j - 3:j + 2] * r_at_g_in_rb_filter
#                 )
#             # Интерполируем синий в позициях зеленого, где в той же строке красный
#             if g_mask[i - 2, j - 2] and r_mask[i - 2, j - 3]:
#                 interpolated_img[i - 2, j - 2, 2] = np.sum(
#                     padded_img[i - 3:i + 2, j - 3:j + 2] * b_at_g_in_rb_filter
#                 )
#             # Интерполируем синий в позициях зеленого, где в той же строке синий
#             if g_mask[i - 2, j - 2] and b_mask[i - 2, j - 1]:
#                 interpolated_img[i - 2, j - 2, 2] = np.sum(
#                     padded_img[i - 3:i + 2, j - 3:j + 2] * b_at_g_in_rb_filter
#                 )
#             # Интерполируем красный в позициях синего
#             if b_mask[i - 2, j - 2]:
#                 interpolated_img[i - 2, j - 2, 0] = np.sum(
#                     padded_img[i - 2:i + 3, j - 2:j + 3] * r_at_b_filter
#                 )
#             # Интерполируем синий в позициях красного
#             if r_mask[i - 2, j - 2]:
#                 interpolated_img[i - 2, j - 2, 2] = np.sum(
#                     padded_img[i - 2:i + 3, j - 2:j + 3] * b_at_r_filter
#                 )

#     # Обрезаем значения пикселей до допустимого диапазона
#     interpolated_img = np.clip(interpolated_img, 0, 255).astype(np.uint8)

#     return interpolated_img

# # def improved_interpolation(raw_img):
# #     """
# #     Args:
# #         raw_img: Массив NumPy, представляющий изображение Байера (одноканальное).

# #     Returns:
# #         Массив NumPy, представляющий интерполированное цветное изображение (трёхканальное).
# #     """
# #     # Переводим изображение в float64 для избежания переполнений
# #     raw_img = raw_img.astype(np.float64)
    
# #     # Размер изображения
# #     height, width = raw_img.shape
    
# #     # Получаем маски для каждого канала
# #     masks = get_bayer_masks(height, width)
# #     r_mask, g_mask, b_mask = masks[..., 0], masks[..., 1], masks[..., 2]
    
# #     # Инициализируем трёхканальное изображение
# #     interpolated_img = np.zeros((height, width, 3), dtype=np.float64)
    
# #     # Заполняем известные значения по маскам
# #     interpolated_img[..., 0][r_mask] = raw_img[r_mask]  # Красный канал
# #     interpolated_img[..., 1][g_mask] = raw_img[g_mask]  # Зелёный канал
# #     interpolated_img[..., 2][b_mask] = raw_img[b_mask]  # Синий канал
    
# #     # Коэффициенты для фильтров (из статьи)
# #     alpha = 1 / 2
# #     beta = 5 / 8
# #     gamma = 3 / 4
    
# #     # Применяем интерполяцию для каждого канала
# #     for i in range(2, height - 2):
# #         for j in range(2, width - 2):
# #             # Интерполяция для зелёного канала в позиции красного пикселя
# #             if r_mask[i, j]:
# #                 g_bilinear = np.mean([interpolated_img[i, j-1, 1], interpolated_img[i, j+1, 1], 
# #                                       interpolated_img[i-1, j, 1], interpolated_img[i+1, j, 1]])  # Зелёный по биллинейной интерполяции
# #                 delta_r = raw_img[i, j] - np.mean([raw_img[i, j-2], raw_img[i, j+2], raw_img[i-2, j], raw_img[i+2, j]])  # Градиент для красного канала
# #                 interpolated_img[i, j, 1] = g_bilinear + alpha * delta_r

# #             # Интерполяция для красного канала в позиции зелёного пикселя
# #             if g_mask[i, j]:
# #                 r_bilinear = np.mean([interpolated_img[i-1, j-1, 0], interpolated_img[i-1, j+1, 0], 
# #                                       interpolated_img[i+1, j-1, 0], interpolated_img[i+1, j+1, 0]])  # Красный по биллинейной интерполяции
# #                 delta_g = raw_img[i, j] - np.mean([raw_img[i-1, j-1], raw_img[i-1, j+1], raw_img[i+1, j-1], raw_img[i+1, j+1]])  # Градиент для зелёного канала
# #                 interpolated_img[i, j, 0] = r_bilinear + beta * delta_g

# #             # Интерполяция для синего канала в позиции зелёного пикселя
# #             if g_mask[i, j]:
# #                 b_bilinear = np.mean([interpolated_img[i-1, j-1, 2], interpolated_img[i-1, j+1, 2], 
# #                                       interpolated_img[i+1, j-1, 2], interpolated_img[i+1, j+1, 2]])  # Синий по биллинейной интерполяции
# #                 delta_g = raw_img[i, j] - np.mean([raw_img[i-1, j-1], raw_img[i-1, j+1], raw_img[i+1, j-1], raw_img[i+1, j+1]])  # Градиент для зелёного канала
# #                 interpolated_img[i, j, 2] = b_bilinear + gamma * delta_g

# #             # Интерполяция для зелёного канала в позиции синего пикселя
# #             if b_mask[i, j]:
# #                 g_bilinear = np.mean([interpolated_img[i, j-1, 1], interpolated_img[i, j+1, 1], 
# #                                       interpolated_img[i-1, j, 1], interpolated_img[i+1, j, 1]])  # Зелёный по биллинейной интерполяции
# #                 delta_b = raw_img[i, j] - np.mean([raw_img[i, j-2], raw_img[i, j+2], raw_img[i-2, j], raw_img[i+2, j]])  # Градиент для синего канала
# #                 interpolated_img[i, j, 1] = g_bilinear + alpha * delta_b

# #     # Приводим значения к диапазону [0, 255] и преобразуем в uint8
# #     interpolated_img = np.clip(interpolated_img, 0, 255).astype(np.uint8)

# #     return interpolated_img

# def improved_interpolation(raw_img):
#     """
#     Args:
#         raw_img: Массив NumPy, представляющий изображение Байера (одноканальное).

#     Returns:
#         Массив NumPy, представляющий интерполированное цветное изображение (трёхканальное).
#     """
#     # Переводим изображение в float64 для избежания переполнений
#     raw_img = raw_img.astype(np.float64)

#     # Размер изображения
#     height, width = raw_img.shape

#     # Получаем маски для каждого канала
#     masks = get_bayer_masks(height, width)
#     r_mask, g_mask, b_mask = masks[..., 0], masks[..., 1], masks[..., 2]

#     # Инициализируем трёхканальное изображение
#     interpolated_img = np.zeros((height, width, 3), dtype=np.float64)

#     # Заполняем известные значения по маскам
#     interpolated_img[..., 0][r_mask] = raw_img[r_mask]  # Красный канал
#     interpolated_img[..., 1][g_mask] = raw_img[g_mask]  # Зелёный канал
#     interpolated_img[..., 2][b_mask] = raw_img[b_mask]  # Синий канал

#     # Коэффициенты для фильтров (из статьи)
#     alpha = 1 / 2
#     beta = 5 / 8
#     gamma = 3 / 4

#     # Применяем интерполяцию для каждого канала
#     for i in range(2, height - 2):
#         for j in range(2, width - 2):
#             # Интерполяция для зелёного канала в позиции красного пикселя
#             if r_mask[i, j]:
#                 g_bilinear = np.mean([interpolated_img[i, j - 1, 1], interpolated_img[i, j + 1, 1],
#                                       interpolated_img[i - 1, j, 1], interpolated_img[i + 1, j, 1]])  # Зелёный по биллинейной интерполяции
#                 delta_r = raw_img[i, j] - np.mean(
#                     [raw_img[i, j - 2], raw_img[i, j + 2], raw_img[i - 2, j],
#                      raw_img[i + 2, j]])  # Градиент для красного канала
#                 interpolated_img[i, j, 1] = g_bilinear + alpha * delta_r

#             # Интерполяция для зелёного канала в позиции синего пикселя
#             if b_mask[i, j]:
#                 g_bilinear = np.mean([interpolated_img[i, j - 1, 1], interpolated_img[i, j + 1, 1],
#                                       interpolated_img[i - 1, j, 1], interpolated_img[i + 1, j, 1]])  # Зелёный по биллинейной интерполяции
#                 delta_b = raw_img[i, j] - np.mean(
#                     [raw_img[i, j - 2], raw_img[i, j + 2], raw_img[i - 2, j],
#                      raw_img[i + 2, j]])  # Градиент для синего канала
#                 interpolated_img[i, j, 1] = g_bilinear + alpha * delta_b

#             # Интерполяция для красного канала в позиции синего пикселя (верхняя левая позиция в квадрате 3x3)
#             if b_mask[i, j] and b_mask[i - 1, j - 1]:
#                 r_bilinear = np.mean([interpolated_img[i - 1, j, 0], interpolated_img[i, j - 1, 0]])  # Красный по биллинейной интерполяции
#                 delta_b = raw_img[i, j] - np.mean(
#                     [raw_img[i, j - 2], raw_img[i - 2, j]])  # Градиент для синего канала
#                 interpolated_img[i - 1, j - 1, 0] = r_bilinear + gamma * delta_b

#             # Интерполяция для красного канала в позиции синего пикселя (нижняя правая позиция в квадрате 3x3)
#             if b_mask[i, j] and b_mask[i + 1, j + 1]:
#                 r_bilinear = np.mean([interpolated_img[i + 1, j, 0], interpolated_img[i, j + 1, 0]])  # Красный по биллинейной интерполяции
#                 delta_b = raw_img[i, j] - np.mean(
#                     [raw_img[i, j + 2], raw_img[i + 2, j]])  # Градиент для синего канала
#                 interpolated_img[i + 1, j + 1, 0] = r_bilinear + gamma * delta_b

#             # Интерполяция для синего канала в позиции красного пикселя (верхняя левая позиция в квадрате 3x3)
#             if r_mask[i, j] and r_mask[i - 1, j - 1]:
#                 b_bilinear = np.mean([interpolated_img[i - 1, j, 2], interpolated_img[i, j - 1, 2]])  # Синий по биллинейной интерполяции
#                 delta_r = raw_img[i, j] - np.mean(
#                     [raw_img[i, j - 2], raw_img[i - 2, j]])  # Градиент для красного канала
#                 interpolated_img[i - 1, j - 1, 2] = b_bilinear + gamma * delta_r

#             # Интерполяция для синего канала в позиции красного пикселя (нижняя правая позиция в квадрате 3x3)
#             if r_mask[i, j] and r_mask[i + 1, j + 1]:
#                 b_bilinear = np.mean([interpolated_img[i + 1, j, 2], interpolated_img[i, j + 1, 2]])  # Синий по биллинейной интерполяции
#                 delta_r = raw_img[i, j] - np.mean(
#                     [raw_img[i, j + 2], raw_img[i + 2, j]])  # Градиент для красного канала
#                 interpolated_img[i + 1, j + 1, 2] = b_bilinear + gamma * delta_r

#             # Интерполяция для красного канала в позиции зелёного пикселя
#             if g_mask[i, j]:
#                 r_bilinear = np.mean([interpolated_img[i - 1, j - 1, 0], interpolated_img[i - 1, j + 1, 0],
#                                       interpolated_img[i + 1, j - 1, 0],
#                                       interpolated_img[i + 1, j + 1, 0]])  # Красный по биллинейной интерполяции
#                 delta_g = raw_img[i, j] - np.mean(
#                     [raw_img[i - 1, j - 1], raw_img[i - 1, j + 1], raw_img[i + 1, j - 1],
#                      raw_img[i + 1, j + 1]])  # Градиент для зелёного канала
#                 interpolated_img[i, j, 0] = r_bilinear + beta * delta_g

#             # Интерполяция для синего канала в позиции зелёного пикселя
#             if g_mask[i, j]:
#                 b_bilinear = np.mean([interpolated_img[i - 1, j - 1, 2], interpolated_img[i - 1, j + 1, 2],
#                                       interpolated_img[i + 1, j - 1, 2],
#                                       interpolated_img[i + 1, j + 1, 2]])  # Синий по биллинейной интерполяции
#                 delta_g = raw_img[i, j] - np.mean(
#                     [raw_img[i - 1, j - 1], raw_img[i - 1, j + 1], raw_img[i + 1, j - 1],
#                      raw_img[i + 1, j + 1]])  # Градиент для зелёного канала
#                 interpolated_img[i, j, 2] = b_bilinear + beta * delta_g

#     # Приводим значения к диапазону [0, 255] и преобразуем в uint8
#     interpolated_img = np.clip(interpolated_img, 0, 255).astype(np.uint8)

#     return interpolated_img

def compute_psnr(img_pred, img_gt):
    """
    Вычисляет метрику PSNR между предсказанным изображением и эталонным.
    
    :param img_pred: np.ndarray - предсказанное изображение (результат интерполяции)
    :param img_gt: np.ndarray - эталонное изображение (ground truth)
    :return: float - значение PSNR
    """
    # Преобразуем изображения в float64 для точности вычислений
    img_pred = img_pred.astype(np.float64)
    img_gt = img_gt.astype(np.float64)
    
    # Вычисляем MSE (среднеквадратичную ошибку)
    mse = np.mean((img_pred - img_gt) ** 2)
    
    # Если MSE равно нулю, то изображения идентичны, выбрасываем исключение
    if mse == 0:
        raise ValueError("MSE равно нулю. Изображения идентичны.")
    
    # Максимальное значение пикселя в эталонном изображении (например, 255 для uint8)
    max_pixel = np.max(img_gt)
    
    # Вычисляем PSNR
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    
    return psnr

