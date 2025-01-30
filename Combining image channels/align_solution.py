import numpy as np
from scipy.signal import correlate2d

def extract_channel_plates(img, crop=False):
    """
    Извлекает цветные пластины из исходного изображения.
    Args:
        img: Одноканальное изображение, полученное сканированием фотопластинки Прокудина-Горского.
        crop: Флаг, указывающий, нужно ли обрезать рамки фотопластинок.
    Returns:
        Кортеж из трех извлеченных каналов – красного, зеленого и синего (в том порядке).
        Кортеж из трех координат, задающих положение левого верхнего угла каждого канала.
    """
    height, width = img.shape
    
    # Определяем высоту каждой пластины
    plate_height = height // 3
    # Удаляем лишние строки, если высота не делится на 3 нацело
    img = img[:plate_height * 3, :]
    
    # Извлекаем каналы
    blue_channel = img[:plate_height, :]
    green_channel = img[plate_height:plate_height * 2, :]
    red_channel = img[plate_height * 2:, :]
    
    # Изначальные координаты левого верхнего угла для каждого канала
    blue_coords = np.array([0, 0])
    green_coords = np.array([plate_height, 0])
    red_coords = np.array([plate_height * 2, 0])
    
    if crop:
        # Определяем ширину рамки в процентах от размера изображения
        crop_height = int(0.1 * plate_height)  # 10% от высоты пластины
        crop_width = int(0.1 * width)          # 10% от ширины изображения

        # Обрезаем каждый канал по 10% с каждой стороны
        blue_channel = blue_channel[crop_height:-crop_height, crop_width:-crop_width]
        green_channel = green_channel[crop_height:-crop_height, crop_width:-crop_width]
        red_channel = red_channel[crop_height:-crop_height, crop_width:-crop_width]

        # Обновляем координаты с учётом обрезки
        blue_coords += np.array([crop_height, crop_width])
        green_coords += np.array([crop_height, crop_width])
        red_coords += np.array([crop_height, crop_width])

    return(red_channel, green_channel, blue_channel), (red_coords, green_coords, blue_coords)

from scipy.ndimage import zoom

def mse(img1, img2):
    return np.mean((img1 - img2) ** 2)
def ncc(img1, img2):
    return np.sum(img1 * img2) / (np.sqrt(np.sum(img1 ** 2)) * np.sqrt(np.sum(img2 ** 2)))

from scipy.fft import fft2, ifft2

def find_relative_shift_fourier(img_a, img_b):
    """
    Находит относительный сдвиг между двумя изображениями с использованием метода кросс-корреляции в частотной области.
    Args:
        img_a: Первое изображение
        img_b: Второе изображение
    Returns:
        Массив NumPy, представляющий сдвиг (dx, dy).
    """
    # Нормализация изображений
    img_a = (img_a - np.mean(img_a)) / np.std(img_a)
    img_b = (img_b - np.mean(img_b)) / np.std(img_b)

    # Вычисление кросс-корреляции в частотной области
    #Здесь выполняется двухмерное быстрое преобразование Фурье (FFT) для изображения img_a и img_b(из пространственной области в частотную)
    f_a = fft2(img_a)
    f_b = fft2(img_b)
    # вычисляется кросс-спектр мощности, путем умножения преобразования Фурье изображения img_a (f_a) на комплексное сопряжение преобразования Фурье изображения img_b (np.conj(f_b)). Комплексное сопряжение меняет знак мнимой части каждого элемента.
    cross_power_spectrum = f_a * np.conj(f_b)
    #обработное быстрое преобразование Фурье
    cross_correlation = ifft2(cross_power_spectrum)

    # Нахождение пика кросс-корреляции
    #np.abs(cross_correlation): Вычисляет абсолютные значения элементов массива cross_correlation.
    #np.unravel_index(..., cross_correlation.shape): Преобразует одномерный индекс в многомерный индекс,
    #соответствующий форме массива cross_correlation
    max_idx = np.unravel_index(np.argmax(np.abs(cross_correlation)), cross_correlation.shape)
    shift_y, shift_x = max_idx

    # Корректировка сдвига
    if shift_x > img_a.shape[1] // 2:
        shift_x -= img_a.shape[1]
    if shift_y > img_a.shape[0] // 2:
        shift_y -= img_a.shape[0]

    return np.array([-shift_y, -shift_x])

def find_absolute_shifts(crops, crop_coords, find_relative_shift_fn):
    """
    Находит абсолютные сдвиги красного и синего каналов по отношению к зеленому.

    Args:
        crops: Кортеж из трех каналов (красный, зеленый, синий).
        crop_coords: Кортеж из трех координат (координаты верхнего левого угла каждого канала).
        find_relative_shift_fn: Функция для нахождения относительных сдвигов.

    Returns:
        Кортеж из двух массивов NumPy, представляющих абсолютные сдвиги красного и синего каналов.
    """
    red_channel, green_channel, blue_channel = crops
    red_coords, green_coords, blue_coords = crop_coords


    # Найти относительные сдвиги
    red_shift = find_relative_shift_fn(green_channel, red_channel)
    blue_shift = find_relative_shift_fn(green_channel, blue_channel)

    # Перевести относительные сдвиги в абсолютные
    r_to_g = red_coords + red_shift - green_coords
    b_to_g = blue_coords + blue_shift - green_coords

    return -r_to_g, -b_to_g

def create_aligned_image(channels, channel_coords, r_to_g, b_to_g):
    """
    Совмещает три канала изображения (красный, зеленый, синий) с учетом абсолютных сдвигов.
    Args:
        channels: Кортеж из трёх каналов (красный, зелёный, синий).
        channel_coords: Кортеж из трёх координат (координаты верхнего левого угла каждого канала).
        r_to_g: Абсолютный сдвиг для красного канала относительно зеленого.
        b_to_g: Абсолютный сдвиг для синего канала относительно зеленого.
    Returns:
        Совмещенное изображение в цвете.
    """
    red_channel, green_channel, blue_channel = channels
    red_coords, green_coords, blue_coords = channel_coords

    # Находим границы перекрывающейся области в абсолютных координатах
    y_0 = max(red_coords[0] + r_to_g[0], green_coords[0], blue_coords[0] + b_to_g[0])
    y_1 = min(red_coords[0]   + r_to_g[0] + red_channel.shape[0], 
              green_coords[0] + green_channel.shape[0], 
              blue_coords[0]  + b_to_g[0] + blue_channel.shape[0])
    
    x_0 = max(red_coords[1] + r_to_g[1], green_coords[1], blue_coords[1] + b_to_g[1])
    x_1 = min(red_coords[1] + r_to_g[1] + red_channel.shape[1],
                green_coords[1] + green_channel.shape[1], 
                blue_coords[1] + b_to_g[1] + blue_channel.shape[1])


     # Создаём пустое изображение для результата
    aligned_image = np.zeros((y_1 - y_0, x_1 - x_0, 3), dtype = np.float64)

    # Вставляем каналы в результирующее изображение с учетом сдвигов
    aligned_image[:, :, 1] = green_channel[y_0 - green_coords[0]:y_1 - green_coords[0],
                                           x_0 - green_coords[1]:x_1 - green_coords[1]]
    aligned_image[:, :, 2] = red_channel[y_0 - red_coords[0] - r_to_g[0]:y_1 - red_coords[0] - r_to_g[0],
                                         x_0 - red_coords[1] - r_to_g[1]:x_1 - red_coords[1] - r_to_g[1]]
    aligned_image[:, :, 0] = blue_channel[y_0 - blue_coords[0] - b_to_g[0]:y_1 - blue_coords[0] - b_to_g[0],
                                          x_0 - blue_coords[1] - b_to_g[1]:x_1 - blue_coords[1] - b_to_g[1]]
    # # Обрезаем каналы по абсолютным координатам
    # green_crop = green_channel[y_0 - green_coords[0]:y_1 - green_coords[0], 
    #                            x_0 - green_coords[1]:x_1 - green_coords[1]]
    # red_crop = red_channel[y_0 - red_coords[0] - r_to_g[0]:y_1 - red_coords[0] - r_to_g[0], 
    #                        x_0 - red_coords[1] - r_to_g[1]:x_1 - red_coords[1] - r_to_g[1]]
    # blue_crop = blue_channel[y_0 - blue_coords[0] - b_to_g[0]:y_1 - blue_coords[0] - b_to_g[0], 
    #                         x_0 - blue_coords[1] - b_to_g[1]:x_1 - blue_coords[1] - b_to_g[1]]

    # # Создаем цветное изображение
    # aligned_image = cv2.merge((blue_crop, green_crop, red_crop))

    return aligned_image

def image_pyramid(img, min_size, scale):
    """
    Создает пирамиду изображений.
    """
    pyramid = [img]
    while img.shape[0] > min_size and img.shape[1] > min_size:
        new_shape = (int(img.shape[0] / scale), int(img.shape[1] / scale))
        img = zoom(img, (new_shape[0] / img.shape[0], new_shape[1] / img.shape[1]))
        pyramid.append(img)
    return pyramid

def find_best_shifts(img_a_i, img_b_i, search_range_x, search_range_y, metric):
    best_score = -float('inf') if metric == 'ncc' else float('inf')
    best_shift = (0, 0)
    for dx in range(*search_range_x):
        for dy in range(*search_range_y):
            shifted_img_b_i = np.roll(img_b_i, (dy, dx), axis=(0, 1))
            
            if metric == 'mse':
                score = mse(img_a_i, shifted_img_b_i)
                if score < best_score:
                    best_score = score
                    best_shift = (dx, dy)
            elif metric == 'ncc':
                score = ncc(img_a_i, shifted_img_b_i)
                if score > best_score:
                    best_score = score
                    best_shift = (dx, dy)

    return best_shift

def find_relative_shift_pyramid(img_a, img_b, metric='mse', max_shift=14, min_size=500, scale=2.0):
    """
    Ищет наилучший сдвиг между двумя изображениями с использованием пирамидального подхода.
    """
    pyramid_a = image_pyramid(img_a, min_size, scale)
    pyramid_b = image_pyramid(img_b, min_size, scale)

    shift_x, shift_y = 0, 0
    for level in reversed(range(len(pyramid_a))):
        img_a_i = pyramid_a[level]
        img_b_i = pyramid_b[level]

        # Уточняем начальное приближение на основе сдвига с предыдущего уровня
        if level < len(pyramid_a) - 1:
            shift_x = int(shift_x * scale)
            shift_y = int(shift_y * scale)
            # На текущем уровне ищем оптимальный сдвиг
            current_max_shift = max(int(max_shift / (scale ** level)), 1)
            search_range_x = (shift_x - current_max_shift, shift_x + current_max_shift + 1)
            search_range_y = (shift_y - current_max_shift, shift_y + current_max_shift + 1)
        else:
            search_range_x = (-max_shift, max_shift + 1)
            search_range_y = (-max_shift, max_shift + 1)

        # Используем функцию find_best_shifts для поиска наилучшего сдвига
        shift_x, shift_y = find_best_shifts(img_a_i, img_b_i, search_range_x, search_range_y, metric)

    return np.array([-shift_y, -shift_x])

if __name__ == "__main__":
    import common
    import pipeline

    # Read the source image and the corresponding ground truth information
    test_path = "tests/05_unittest_align_image_pyramid_img_small_input/00"
    raw_img, (r_point, g_point, b_point) = common.read_test_data(test_path)

    # Draw the same point on each channel in the original
    # raw image using the ground truth coordinates
    visualized_img = pipeline.visualize_point(raw_img, r_point, g_point, b_point)
    common.save_image(f"gt_visualized.png", visualized_img)

    for method in ["pyramid", "fourier"]:
        # Run the whole alignment pipeline
        r_to_g, b_to_g, aligned_img = pipeline.align_image(raw_img, method)
        common.save_image(f"{method}_aligned.png", aligned_img)

        # Draw the same point on each channel in the original
        # raw image using the predicted r->g and b->g shifts
        # (Compare with gt_visualized for debugging purposes)
        r_pred = g_point - r_to_g
        b_pred = g_point - b_to_g
        visualized_img = pipeline.visualize_point(raw_img, r_pred, g_point, b_pred)

        r_error = abs(r_pred - r_point)
        b_error = abs(b_pred - b_point)
        print(f"{method}: {r_error = }, {b_error = }")

        common.save_image(f"{method}_visualized.png", visualized_img)