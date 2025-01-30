import numpy as np
from scipy.ndimage import convolve

def copy_arr(pattern, shape):
    tile = np.tile(pattern, np.array(shape) // 2 + np.array(shape) % 2)
    if shape[0] % 2:
        tile = tile[:-1, :]
    if shape[1] % 2:
        tile = tile[:, :-1]
    return tile.astype(bool)
    
def get_bayer_masks(n_rows, n_cols):
    red_mask = np.zeros((n_rows, n_cols), dtype = bool)
    green_mask = np.zeros((n_rows, n_cols), dtype = bool)
    blue_mask = np.zeros((n_rows, n_cols), dtype = bool)

    red_mask[::2, 1::2] = True
    blue_mask[1::2, ::2] = True
    green_mask[::2, ::2] = True
    green_mask[1::2, 1::2] = True

    masks = np.stack((red_mask, green_mask, blue_mask), axis =- 1)
    return masks

def get_colored_img(raw_img):

    
    n_rows, n_cols = raw_img.shape
    masks = get_bayer_masks(n_rows, n_cols)

    # Инициализируем трехканальное изображение
    colored_img = np.zeros((n_rows, n_cols, 3), dtype = raw_img.dtype)

    # Заполняем цветовые каналы согласно маскам
    for channel in range(3):
        colored_img[:, :, channel] = np.where(masks[:, :, channel], raw_img, 0)

    return colored_img

def get_raw_img(colored_img):
    n_rows, n_cols, _ = colored_img.shape
    masks = get_bayer_masks(n_rows, n_cols)

    # Инициализируем одноканальное изображение
    raw_img = np.zeros((n_rows, n_cols), dtype = colored_img.dtype)

    # Заполняем одноканальное изображение согласно маскам
    for channel in range(3):
        raw_img += np.where(masks[:, :, channel], colored_img[:, :, channel], 0)

    return raw_img

def bilinear_interpolation(raw_img):
    n_rows, n_cols = raw_img.shape
    rgb_img = np.zeros((n_rows, n_cols, 3), dtype=np.float64)
    
    # Создание масок для каждого канала
    red_mask = np.zeros_like(raw_img, dtype = bool)
    blue_mask = np.zeros_like(raw_img, dtype = bool)
    green_mask = np.zeros_like(raw_img, dtype = bool)
    
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
        known_values = convolve(channel * mask, kernel, mode='constant', cval = 0.0)
        counts = convolve(mask.astype(np.float64), kernel, mode='constant', cval = 0.0)
        # Вычисление среднего значения
        interpolated_values = known_values / np.maximum(counts, 1)
        channel[~mask] = interpolated_values[~mask]
    
    # Интерполяция для каждого канала
    interpolate_channel(rgb_img[:, :, 0], red_mask)
    interpolate_channel(rgb_img[:, :, 1], green_mask)
    interpolate_channel(rgb_img[:, :, 2], blue_mask)
    
    # Приведение значений к диапазону [0, 255] 
    rgb_img = np.clip(rgb_img, 0, 255).astype(np.uint8)
    
    return rgb_img

def improved_interpolation(raw_img):
    #нормированнные фильтры из статьи
    G_at_R_filter = np.array([
        [ 0,  0, -1,  0,  0],
        [ 0,  0,  2,  0,  0],
        [-1,  2,  4,  2, -1],
        [ 0,  0,  2,  0,  0],
        [ 0,  0, -1,  0,  0]
    ], dtype=np.float64) / 8

    R_at_G_R_row_B_col_filter = np.array([
        [ 0,  0, 1/2,  0,  0],
        [ 0, -1,   0, -1,  0],
        [-1,  4,   5,  4, -1],
        [ 0, -1,   0, -1,  0],
        [ 0,  0, 1/2,  0,  0]
    ], dtype=np.float64) / 8

    R_at_G_B_row_R_col_filter = np.array([
        [  0,  0,  -1,  0,   0],
        [  0, -1,   4, -1,   0],
        [1/2,  0,   5,  0, 1/2],
        [  0, -1,   4, -1,   0],
        [  0,  0,  -1,  0,   0]
    ], dtype=np.float64) / 8

    R_at_B_filter = np.array([
        [   0,  0, -3/2,  0,    0],
        [   0,  2,    0,  2,    0],
        [-3/2,  0,    6,  0, -3/2],
        [   0,  2,    0,  2,    0],
        [   0,  0, -3/2,  0,    0]
    ], dtype=np.float64) / 8

    G_at_B_filter = G_at_R_filter
    B_at_G_B_row_R_col_filter = R_at_G_R_row_B_col_filter
    B_at_G_R_row_B_col_filter = R_at_G_B_row_R_col_filter
    B_at_R_filter = R_at_B_filter
    
    #маски для выделения пикселей в нужных позициях
    G_at_R_mask = copy_arr([[0, 1], 
                              [0, 0]], raw_img.shape[:2])
    G_at_B_mask = copy_arr([[0, 0], 
                              [1, 0]], raw_img.shape[:2])
    R_at_G_RB_mask = copy_arr([[1, 0],
                                 [0, 0]], raw_img.shape[:2])
    R_at_G_BR_mask = copy_arr([[0, 0],
                                 [0, 1]], raw_img.shape[:2])
    G_mask = copy_arr([[1, 0], 
                         [0, 1]], raw_img.shape[:2])

    R_B_mask = G_at_B_mask
    B_at_G_BR_mask = R_at_G_BR_mask
    B_at_G_RB_mask = R_at_G_RB_mask
    B_R_mask = G_at_R_mask
    
    #подсчет с учетом масок и фильтров
    raw_img = raw_img.astype(np.int32)
    R = convolve(raw_img, R_at_G_R_row_B_col_filter) * R_at_G_RB_mask + convolve(raw_img, R_at_G_B_row_R_col_filter) * R_at_G_BR_mask + convolve(raw_img, R_at_B_filter) * R_B_mask + raw_img * G_at_R_mask
    G = convolve(raw_img, G_at_R_filter) * G_at_R_mask + convolve(raw_img, G_at_B_filter) * G_at_B_mask + raw_img * G_mask
    B = convolve(raw_img, B_at_G_B_row_R_col_filter) * B_at_G_BR_mask + convolve(raw_img, B_at_G_R_row_B_col_filter) * B_at_G_RB_mask + convolve(raw_img, B_at_R_filter) * B_R_mask + raw_img * G_at_B_mask
    
    return np.dstack((R, G, B)).clip(0, 255).astype(np.uint8)

def compute_psnr(img_pred, img_gt):
    """
    Вычисляет метрику PSNR между предсказанным изображением и эталонным.
    
    :param img_pred: np.ndarray - предсказанное изображение (результат интерполяции)
    :param img_gt: np.ndarray - эталонное изображение (ground truth)
    :return: float - значение PSNR
    """
    if img_pred.ndim != 3 or img_gt.ndim != 3:
      raise ValueError("Ожидаются трёхканальные изображения")
    if img_pred.dtype != np.uint8 or img_gt.dtype != np.uint8:
        raise ValueError("Ожидаются изображения типа uint8")
    # Преобразуем изображения в float64 для точности вычислений
    img_pred = img_pred.astype(np.float64)
    img_gt = img_gt.astype(np.float64)
    mse = np.mean((img_pred - img_gt) ** 2)
    # Максимальное значение пикселя в эталонном изображении (например, 255 для uint8)
    max_pixel = np.max(img_gt)
    if mse == 0:
        raise(ValueError)
    # Вычисляем PSNR
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr

