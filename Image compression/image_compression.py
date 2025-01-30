import numpy as np
import io
import pickle
import zipfile

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio

def pca_compression(matrix, p):
     
    """Сжатие изображения с помощью PCA
        Вход: двумерная матрица (одна цветовая компонента картинки), количество компонент
        Выход: собственные векторы, проекция матрицы на новое пр-во и средние значения до центрирования
    """
    # 1. Центрирование каждой строчки матрицы, вычисляет среднее значение для каждой строки, потом приводим к двумерному массиву
    mean = np.mean(matrix, axis = 1)
    centered_matrix = matrix - mean[:, None]

    # 2. Поиск матрицы ковариации
    cov_matrix = np.cov(centered_matrix)

    # 3. Поиск собственных значений и собственных векторов
    eig_val, eig_vec = np.linalg.eigh(cov_matrix)

    # 4. Сортировка собственных значений и соответствующих им собственных векторов
        #возвращает индексы, которые отсортировали бы массив по возрастанию, но потом срез переворачивает массив, и получаем по убываю+нию
    sorted_indices = np.argsort(eig_val)[::-1]
    sorted_eig_val = eig_val[sorted_indices]
        #берем все строки и сортируем по столбцам
    sorted_eig_vec = eig_vec[:, sorted_indices] 

    # 5. Оставляем только p первых собственных векторов
    selected_eig_vec = sorted_eig_vec[:, :p]

    # 6. Проекция данных на новое пространство
        #dot - матричное умножение двух массивов, selected_eig_vec.T - транспонированная матрица z = x^T * W, используем SVD разложение
    projected_matrix = np.dot(selected_eig_vec.T, centered_matrix)

    return selected_eig_vec, projected_matrix, mean


def pca_decompression(compressed):
    """Разжатие изображения
        Вход: список кортежей из 3 собственных векторов и проекций для каждой цветовой компоненты
        Выход: разжатое изображение
    """
    decompressed_channels = []
    
    for eig_vec, projections, mean in compressed:
        # Восстановление центрированной матрицы
        centered_matrix = np.dot(eig_vec, projections)
        
        # Восстановление исходной матрицы
        original_matrix = centered_matrix + mean[:, None]
        decompressed_channels.append(original_matrix)
    
    decompressed_image = np.stack(decompressed_channels, axis=-1)
    
    return np.clip(decompressed_image, 0, 255)

def pca_visualize():
    plt.clf()
    img = imread("cat.png")
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(3, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 5, 10, 20, 50, 100, 150, 200, 256]):
        compressed = []
        for j in range(3):
            channel = img[..., j].astype(np.float64)
            compressed.append(pca_compression(channel, p))

        decompressed_img = pca_decompression(compressed)
        decompressed_img = np.clip(decompressed_img, 0, 255).astype(np.uint8)

        axes[i // 3, i % 3].imshow(decompressed_img)
        axes[i // 3, i % 3].set_title("Компонент: {}".format(p))
        axes[i // 3, i % 3].axis('off')

    fig.savefig("pca_visualization.png")




def rgb2ycbcr(image):
    """Переход из пр-ва RGB в пр-во YCbCr
        Вход: RGB изображение
        Выход: YCbCr изображение
    """
    # Матрица преобразования
    transform_matrix = np.array([[  0.299,   0.587,   0.114],
                                 [-0.1687, -0.3313,     0.5],
                                 [    0.5, -0.4187, -0.0813]])
    
    ycbcr = np.dot(image, transform_matrix.T)
    ycbcr += [0, 128, 128]
    
    return ycbcr

def ycbcr2rgb(image):
    """Переход из пр-ва YCbCr в пр-во RGB
        Вход: YCbCr изображение
        Выход: RGB изображение
    """
    # Матрица обратного преобразования
    inverse_transform_matrix = np.array([[ 1,        0,    1.402],
                                         [ 1, -0.34414, -0.71414],
                                         [ 1,     1.77,        0]])
    
    rgb = np.dot(image - [0, 128, 128], inverse_transform_matrix.T)

    return np.clip(rgb, 0, 255)


def get_gauss_1():
    plt.clf()
    rgb_img = imread("Lenna.png")
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[:, :, :3]

    ycbcr_img = rgb2ycbcr(rgb_img)
    
    # Размытие цветовых компонентов Cb и Cr
    ycbcr_img[:, :, 1] = gaussian_filter(ycbcr_img[:, :, 1], sigma = 10)
    ycbcr_img[:, :, 2] = gaussian_filter(ycbcr_img[:, :, 2], sigma = 10)
    
    # Преобразование обратно в RGB
    result_img = ycbcr2rgb(ycbcr_img)
    result_img = np.clip(result_img, 0, 255).astype(np.uint8)

    plt.imshow(result_img)
    plt.axis('off')
    plt.savefig("gauss_1.png")

def get_gauss_2():
    plt.clf()
    rgb_img = imread("Lenna.png")
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[:, :, :3]

    ycbcr_img = rgb2ycbcr(rgb_img)
    
    # Размытие компоненты яркости Y
    ycbcr_img[:, :, 0] = gaussian_filter(ycbcr_img[:, :, 0], sigma = 10)
    
    # Преобразование обратно в RGB
    result_img = ycbcr2rgb(ycbcr_img)
    result_img = np.clip(result_img, 0, 255).astype(np.uint8)

    plt.imshow(result_img)
    plt.axis('off')
    plt.savefig("gauss_2.png")


def downsampling(component):
    """Уменьшаем цветовые компоненты в 2 раза
        Вход: цветовая компонента размера [A, B]
        Выход: цветовая компонента размера [A // 2, B // 2]
    """
    blurred_component = gaussian_filter(component, sigma = 10)
    downsampled_component = blurred_component[::2, ::2]
    
    return downsampled_component


def alpha(c):
    return 1 / np.sqrt(2) if c == 0 else 1

def dct(block):
    """Дискретное косинусное преобразование
        Вход: блок размера 8x8
        Выход: блок размера 8x8 после ДКП
    """
    N = 8
    result = np.zeros((N, N))
    
    for u in range(N):
        for v in range(N):
            sum = 0
            for x in range(N):
                for y in range(N):
                    sum += block[x, y] * np.cos((2 * x + 1) * u * np.pi / (2 * N)) * np.cos((2 * y + 1) * v * np.pi / (2 * N))
            result[u, v] = (1 / 4) * alpha(u) * alpha(v) * sum
    return result

# Матрица квантования яркости
y_quantization_matrix = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ]
)

# Матрица квантования цвета
color_quantization_matrix = np.array(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ]
)

def quantization(block, quantization_matrix):
    """Квантование
        Вход: блок размера 8x8 после применения ДКП; матрица квантования
        Выход: блок размера 8x8 после квантования. Округление осуществляем с помощью np.round
    """
    quantized_block = np.round(block / quantization_matrix)
    return quantized_block

import numpy as np

def calculate_scale_factor(q):
    """Вычисляет Scale Factor S на основе Quality Factor Q"""
    if q < 50:
        S = 5000 / q
    elif 50 <= q < 100:
        S = 200 - 2 * q
    else:
        S = 1
    return S

def own_quantization_matrix(default_quantization_matrix, q):
    """Генерация собственной матрицы квантования на основе заданного Quality Factor"""
    assert 1 <= q <= 100
    S = calculate_scale_factor(q)
    # Создаем новую матрицу квантования с пересчитанными значениями
    new_quantization_matrix = np.floor((50 + S * default_quantization_matrix) / 100).astype(int)
    # Заменяем нули единицами
    new_quantization_matrix[new_quantization_matrix == 0] = 1
    return new_quantization_matrix


def zigzag(block):
    """Зигзаг-сканирование
        Вход: блок размера 8x8
        Выход: список из элементов входного блока, получаемый после его обхода зигзаг-сканированием
    """
    result = []
    n = 8  # Размер блока
    for i in range(2 * n - 1):  # 2*n - 1 диагоналей
        if i % 2 == 0:  # Четные диагонали
            row = min(i, n - 1)
            col = i - row
            while row >= 0 and col < n:
                result.append(block[row][col])
                row -= 1
                col += 1
        else:  # Нечетные диагонали
            col = min(i, n - 1)
            row = i - col
            while col >= 0 and row < n:
                result.append(block[row][col])
                row += 1
                col -= 1
    return result

def compression(zigzag_list):
    """Сжатие последовательности после зигзаг-сканирования
        Вход: список после зигзаг-сканирования
        Выход: сжатый список в формате, который был приведен в качестве примера в самом начале данного пункта
    """
    compressed = []
    zero_count = 0

    for value in zigzag_list:
        if value == 0:
            zero_count += 1  # Увеличиваем счетчик нулей
        else:
            if zero_count > 0:
                compressed.append(0)  # Добавляем один ноль
                compressed.append(zero_count)  # Добавляем количество нулей
                zero_count = 0  # Сбрасываем счетчик
            compressed.append(value)  # Добавляем текущее ненулевое значение

    # Если в конце списка были нули, добавляем их
    if zero_count > 0:
        compressed.append(0)
        compressed.append(zero_count)

    return compressed

def block_process(block, quant_matrix):
    """Обработка блока: DCT, квантование, зигзаг-сканирование и сжатие"""
    # Применяем DCT
    dct_block = dct(block)
    # Применяем квантование
    quant_block = quantization(dct_block, quant_matrix)
    # Применяем зигзаг-сканирование
    zigzag_block = zigzag(quant_block)
    # Применяем сжатие (например, RLE)
    compressed_block = compression(zigzag_block)
    return compressed_block


def jpeg_compression(img, quantization_matrixes):
    """
    Сжатие изображения алгоритмом JPEG
    Вход: цветная картинка img и список из 2-ух матриц квантования quantization_matrixes
    Выход: список списков со сжатыми векторами: [[compressed_y1,...], [compressed_Cb1,...], [compressed_Cr1,...]]
    """
    # 1. Преобразование изображения в YCbCr
    ycbcr_img = rgb2ycbcr(img)
    
    # 2. Уменьшение цветовых компонентов
    downsampled_cb = downsampling(ycbcr_img[:, :, 1])
    downsampled_cr = downsampling(ycbcr_img[:, :, 2])
    
    # 3. Деление на блоки 8x8 и перевод элементов блоков из [0, 255] в [-128, 127]
    height, width, _ = ycbcr_img.shape
    compressed_y = []
    compressed_cb = []
    compressed_cr = []
    
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            # Обработка блока Y
            y_block = ycbcr_img[i:i+8, j:j+8, 0] - 128
            compressed_y.append(block_process(y_block, quantization_matrixes[0]))
    
    for i in range(0, height//2, 8):
        for j in range(0, width//2, 8):
            # Обработка блока Cb
            cb_block = downsampled_cb[i:i+8, j:j+8] - 128
            compressed_cb.append(block_process(cb_block, quantization_matrixes[1]))
            
            # Обработка блока Cr
            cr_block = downsampled_cr[i:i+8, j:j+8] - 128
            compressed_cr.append(block_process(cr_block, quantization_matrixes[1]))


    return [compressed_y, compressed_cb, compressed_cr]


def inverse_compression(compressed_list):
    """Разжатие последовательности
        Вход: сжатый список
        Выход: разжатый список
    """
    decompressed_list = []
    i = 0
    while i < len(compressed_list):
        if compressed_list[i] == 0 and i + 1 < len(compressed_list):
            decompressed_list.extend([0] * compressed_list[i + 1])
            i += 2
        else:
            decompressed_list.append(compressed_list[i])
            i += 1
    return decompressed_list


def inverse_zigzag(input):
    """Обратное зигзаг-сканирование
    Вход: список элементов
    Выход: блок размера 8x8 из элементов входного списка, расставленных в матрице в порядке их следования в зигзаг-сканировании
    """

    n = 8
    output = [[0] * n for _ in range(n)]
    index = 0
    for i in range(2 * n - 1):
        if i % 2 == 0:
            row = min(i, n - 1)
            col = i - row
            while row >= 0 and col < n:
                output[row][col] = input[index]
                index += 1
                row -= 1
                col += 1
        else:
            col = min(i, n - 1)
            row = i - col
            while col >= 0 and row < n:
                output[row][col] = input[index]
                index += 1
                row += 1
                col -= 1
    return output

def inverse_quantization(block, quantization_matrix):
    """Обратное квантование
        Вход: блок размера 8x8 после применения обратного зигзаг-сканирования; матрица квантования
        Выход: блок размера 8x8 после квантования. Округление не производится
    """
    # Умножаем каждый элемент блока на соответствующий элемент матрицы квантования
    dequantized_block = block * quantization_matrix
    
    return dequantized_block

def inverse_dct(block):
    """Обратное дискретное косинусное преобразование
        Вход: блок размера 8x8
        Выход: блок размера 8x8 после обратного ДКП. Округление осуществляется с помощью np.round
    """
    n = 8
    result = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            sum_value = 0
            for x in range(n):
                for y in range(n):
                    sum_value += alpha(x) * alpha(y) * block[x, y] * \
                                 np.cos((2 * i + 1) * x * np.pi / (2 * n)) * \
                                 np.cos((2 * j + 1) * y * np.pi / (2 * n))
            result[i, j] = sum_value / 4
    
    # Округляем результат до ближайшего целого числа
    return np.round(result).astype(int)

def upsampling(component):
    """Увеличивает цветовые компоненты в 2 раза.
       Вход: цветовая компонента размера [A, B] или [A, B, 1]
       Выход: цветовая компонента размера [2 * A, 2 * B, 1]
    """
    if component.ndim == 2:
        component = component[:, :, np.newaxis]
    A, B, _ = component.shape
    upsampled_component = np.zeros((2 * A, 2 * B, 1), dtype=component.dtype)

    for i in range(A):
        for j in range(B):
            upsampled_component[2 * i, 2 * j, 0] = component[i, j, 0]
            upsampled_component[2 * i, 2 * j + 1, 0] = component[i, j, 0]
            upsampled_component[2 * i + 1, 2 * j, 0] = component[i, j, 0]
            upsampled_component[2 * i + 1, 2 * j + 1, 0] = component[i, j, 0]
    return upsampled_component



def jpeg_decompression(result, result_shape, quantization_matrices):
    """Разжатие изображения
        Вход: result список сжатых данных, размер ответа, список из 2-ух матриц квантования
        Выход: разжатое изображение
    """
    height, width = result_shape[:2] 
    Y_blocks, Cb_blocks, Cr_blocks  = [], [], []
    Y, Cb, Cr = [], [], []

    for k, components in enumerate(result):
        for comp in components:
            q_i = 0 if k == 0 else 1
            block = inverse_compression(comp)
            block = inverse_zigzag(block)
            block = inverse_quantization(block, quantization_matrices[q_i])
            block = inverse_dct(block) + 128
            if k == 0:
                Y_blocks.append(block)
            elif k == 1:
                Cb_blocks.append(block)
            else:
                Cr_blocks.append(block)
    # собираем блоки в компоненты    
    block_count = 0
    for i in range(height // 8):
        Y.append([])
        for j in range(width // 8):
            Y[i].append(Y_blocks[block_count])
            block_count+= 1
    
    block_count= 0
    for i in range(height // 16):
        Cb.append([])
        for j in range(width // 16):
            Cb[i].append(Cb_blocks[block_count])
            block_count+= 1
    
    block_count= 0
    for i in range(height // 16):
        Cr.append([])
        for j in range(width // 16):
            Cr[i].append(Cr_blocks[block_count])
            block_count+= 1

    Cb = upsampling(np.block(Cb)[..., np.newaxis])[..., 0]        
    Cr = upsampling(np.block(Cr)[..., np.newaxis])[..., 0]
    Y = np.block(Y)

    ycbcr_img = ycbcr2rgb(np.dstack([Y, Cb, Cr]))
    
    return np.clip(ycbcr_img, 0, 255).astype(np.uint8)


def jpeg_visualize():
    plt.clf()
    img = imread('Lenna.png')
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(2, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 10, 20, 50, 80, 100]):
        quantization_matrices = [own_quantization_matrix(y_quantization_matrix, p), own_quantization_matrix(color_quantization_matrix, p)]
        compressed = jpeg_compression(img, quantization_matrices)
        decompressed = jpeg_decompression(compressed, img.shape, quantization_matrices)  # Передаем только высоту и ширину
        axes[i // 3, i % 3].imshow(decompressed)
        axes[i // 3, i % 3].set_title('Quality Factor: {}'.format(p))

    fig.savefig("jpeg_visualization.png")


def get_deflated_bytesize(data):
    raw_data = pickle.dumps(data)
    with io.BytesIO() as buf:
        with (
            zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zipf,
            zipf.open("data", mode="w") as handle,
        ):
            handle.write(raw_data)
            handle.flush()
            handle.close()
            zipf.close()
        buf.flush()
        return buf.getbuffer().nbytes


def compression_pipeline(img, c_type, param=1):
    """Pipeline для PCA и JPEG
    Вход: исходное изображение; название метода - 'pca', 'jpeg';
    param - кол-во компонент в случае PCA, и Quality Factor для JPEG
    Выход: изображение; количество бит на пиксель
    """

    assert c_type.lower() == "jpeg" or c_type.lower() == "pca"

    if c_type.lower() == "jpeg":
        y_quantization = own_quantization_matrix(y_quantization_matrix, param)
        color_quantization = own_quantization_matrix(color_quantization_matrix, param)
        matrixes = [y_quantization, color_quantization]

        compressed = jpeg_compression(img, matrixes)
        img = jpeg_decompression(compressed, img.shape, matrixes)
        compressed_size = get_deflated_bytesize(compressed)

    elif c_type.lower() == "pca":
        compressed = [
            pca_compression(c.copy(), param)
            for c in img.transpose(2, 0, 1).astype(np.float64)
        ]

        img = pca_decompression(compressed)
        compressed_size = sum(d.nbytes for c in compressed for d in c)

    raw_size = img.nbytes

    return img, compressed_size / raw_size


def calc_metrics(img_path, c_type, param_list):
    """Подсчет PSNR и Compression Ratio для PCA и JPEG. Построение графиков
    Вход: пусть до изображения; тип сжатия; список параметров: кол-во компонент в случае PCA, и Quality Factor для JPEG
    """

    assert c_type.lower() == "jpeg" or c_type.lower() == "pca"

    img = imread(img_path)
    if len(img.shape) == 3:
        img = img[..., :3]

    outputs = []
    for param in param_list:
        outputs.append(compression_pipeline(img.copy(), c_type, param))

    psnr = [peak_signal_noise_ratio(img, output[0]) for output in outputs]
    ratio = [output[1] for output in outputs]

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    fig.set_figwidth(20)
    fig.set_figheight(5)

    ax1.set_title("Quality Factor vs PSNR for {}".format(c_type.upper()))
    ax1.plot(param_list, psnr, "tab:orange")
    ax1.set_ylim(13, 64)
    ax1.set_xlabel("Quality Factor")
    ax1.set_ylabel("PSNR")

    ax2.set_title("PSNR vs Compression Ratio for {}".format(c_type.upper()))
    ax2.plot(psnr, ratio, "tab:red")
    ax2.set_xlim(13, 30)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("PSNR")
    ax2.set_ylabel("Compression Ratio")
    return fig


def get_pca_metrics_graph():
    plt.clf()
    fig = calc_metrics("Lenna.png", "pca", [1, 5, 10, 20, 50, 100, 150, 200, 256])
    fig.savefig("pca_metrics_graph.png")


def get_jpeg_metrics_graph():
    plt.clf()
    fig = calc_metrics("Lenna.png", "jpeg", [1, 10, 20, 50, 80, 100])
    fig.savefig("jpeg_metrics_graph.png")

if __name__ == "__main__":
    pca_visualize()
    get_gauss_1()
    get_gauss_2()
    jpeg_visualize()
    get_pca_metrics_graph()
    get_jpeg_metrics_graph()