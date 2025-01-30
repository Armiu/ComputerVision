from interface import *


# ================================= 1.4.1 SGD ================================
class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def get_parameter_updater(self, parameter_shape):
        """
        :param parameter_shape: tuple, the shape of the associated parameter

        :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
            :param parameter: np.array, current parameter values
            :param parameter_grad: np.array, current gradient, dLoss/dParam

            :return: np.array, new parameter values
            """
            # your code here \/
            return parameter - self.lr * parameter_grad
            # your code here /\

        return updater


# ============================= 1.4.2 SGDMomentum ============================
class SGDMomentum(Optimizer):
    def __init__(self, lr, momentum=0.0):
        self.lr = lr
        self.momentum = momentum

    def get_parameter_updater(self, parameter_shape):
        """
        :param parameter_shape: tuple, the shape of the associated parameter

        :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
            :param parameter: np.array, current parameter values
            :param parameter_grad: np.array, current gradient, dLoss/dParam

            :return: np.array, new parameter values
            """
            # your code here \/

            # Обновляем тензор инерции
            updater.inertia = self.momentum * updater.inertia + self.lr * parameter_grad
            # Обновляем параметры с учетом инерции
            new_parameter = parameter - updater.inertia
            return new_parameter
        
            # your code here /\

        updater.inertia = np.zeros(parameter_shape)
        return updater


# ================================ 2.1.1 ReLU ================================
class ReLU(Layer):
    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, ...)), input values

        :return: np.array((n, ...)), output values

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        self.forward_inputs = np.copy(inputs)
        #заменяем все отрицательные значения нулями
        return np.maximum(inputs, 0)
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

        :return: np.array((n, ...)), dLoss/dInputs

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        grad_inputs = np.where(self.forward_inputs >= 0, grad_outputs, 0)
        return grad_inputs
        # your code here /\


# =============================== 2.1.2 Softmax ==============================
class Softmax(Layer):
    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d)), input values

        :return: np.array((n, d)), output values

            n - batch size
            d - number of units
        """
        # your code here \/
        self.forward_inputs = np.copy(inputs)
        #нормализуем
        norm_exp_inputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) 
        return norm_exp_inputs / np.sum(norm_exp_inputs, axis=1, keepdims=True) #self.forward_outputs
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, d)), dLoss/dOutputs

        :return: np.array((n, d)), dLoss/dInputs

            n - batch size
            d - number of units
        """
        # your code here \/
        # вычисляет сумму произведений по каждой строке, что эквивалентно внешнему произведению вектора y с самим собой.
        # grad_outputs - np.sum(grad_outputs * self.forward_outputs, axis=1, keepdims=True) 
        # соответствует выражению d_ik - y_k, d_ik - символ кронекера
        # self.forward_outputs * grad_outputs = Y * dL/dY , здесь * - поэлементное умножение
        #- np.sum(grad_outputs * self.forward_outputs, axis=1, keepdims=True) = -Y * Y^T * dL/dY
        grad_inputs = self.forward_outputs * (grad_outputs - np.sum(grad_outputs * self.forward_outputs, axis=1, keepdims=True))
        return grad_inputs
        
        # softmax_outputs = self.forward_outputs
        # batch_size, num_classes = softmax_outputs.shape
        # grad_inputs = np.zeros_like(grad_outputs)

        # for i in range(batch_size):
        #     y = softmax_outputs[i].reshape(-1, 1) 
        #     jacobian = np.diagflat(y) - np.dot(y, y.T)
        #     grad_inputs[i] = np.dot(jacobian, grad_outputs[i])

        # return grad_inputs
        # your code here /\


# ================================ 2.1.3 Dense ===============================
class Dense(Layer):
    def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_units = units

        self.weights, self.weights_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        (input_units,) = self.input_shape
        output_units = self.output_units

        # Register weights and biases as trainable parameters
        # Note, that the parameters and gradients *must* be stored in
        # self.<p> and self.<p>_grad, where <p> is the name specified in
        # self.add_parameter

        self.weights, self.weights_grad = self.add_parameter(
            name="weights",
            shape=(output_units, input_units),
            initializer=he_initializer(input_units),
        )

        self.biases, self.biases_grad = self.add_parameter(
            name="biases",
            shape=(output_units,),
            initializer=np.zeros,
        )

        self.output_shape = (output_units,)

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d)), input values

        :return: np.array((n, c)), output values

            n - batch size
            d - number of input units
            c - number of output units
        """
        # your code here \/
        self.forward_inputs = inputs
        dense_outputs = np.dot(inputs, self.weights.T) + self.biases
        return dense_outputs
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, c)), dLoss/dOutputs

        :return: np.array((n, d)), dLoss/dInputs

            n - batch size
            d - number of input units
            c - number of output units
        """
        # your code here \/
        #grad_outputs = dL/dY (градиент по выходам), self.forward_inputs = X (входных данным)
        self.weights_grad = np.dot(grad_outputs.T, self.forward_inputs) #dL/dW=dL/dY * X^T
        #суммируем градиенты по выходам по оси батча (по каждой компоненте отдельно), чтобы получить градиент по смещениям 
        self.biases_grad = np.sum(grad_outputs, axis=0) #dL/dB = dL/dY
        grad_inputs = np.dot(grad_outputs, self.weights) #dL/dX = W^T * dL/dY

        return grad_inputs
        # your code here /\


# ============================ 2.2.1 Crossentropy ============================
class CategoricalCrossentropy(Loss):
    def value_impl(self, y_gt, y_pred):
        """
        :param y_gt: np.array((n, d)), ground truth (correct) labels
        :param y_pred: np.array((n, d)), estimated target values

        :return: np.array((1,)), mean Loss scalar for batch

            n - batch size
            d - number of units
        """
        # your code here \/
        #y_gt - правильный результат(эталлоный) для входа X
        #y_pred - предсказание нейронной сети
        return np.array([-np.mean(np.sum(y_gt * np.log(np.clip(y_pred, eps, None)), axis=1))])
        # your code here /\

    def gradient_impl(self, y_gt, y_pred):
        """
        :param y_gt: np.array((n, d)), ground truth (correct) labels
        :param y_pred: np.array((n, d)), estimated target values

        :return: np.array((n, d)), dLoss/dY_pred

            n - batch size
            d - number of units
        """
        # your code here \/    
        y_pred = np.clip(y_pred, eps, None)
        return -y_gt / (y_pred * y_pred.shape[0])
        # your code here /\


# ======================== 2.3 Train and Test on MNIST =======================
def train_mnist_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    loss = CategoricalCrossentropy()
    #обновление весов на основе функции потерь, lr - шаг обучения
    optimizer = SGD(lr=0.05)
    model_mnist = Model(loss=loss, optimizer=optimizer)

    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)
    #units - количество нейронов в модели
    model_mnist.add(Dense(units=1024, input_shape=(784,)))
    #применяется к выходу предыдущего слоя
    model_mnist.add(ReLU())
    model_mnist.add(Dense(units=512))
    model_mnist.add(ReLU())
    model_mnist.add(Dense(units=128))
    model_mnist.add(ReLU())
    model_mnist.add(Dense(units=10))
    model_mnist.add(Softmax())

    print(model_mnist)

    # 3) Train and validate the model using the provided data
    
    #количество эпох 6
    model_mnist.fit(x_train, y_train, epochs=10, batch_size=32, x_valid=x_valid, y_valid=y_valid)

    # your code here /\
    return model_mnist


# ============================== 3.3.2 convolve ==============================
def convolve(inputs, kernels, padding=0):
    """
    :param inputs: np.array((n, d, ih, iw)), input values
    :param kernels: np.array((c, d, kh, kw)), convolution kernels
    :param padding: int >= 0, the size of padding, 0 means 'valid'

    :return: np.array((n, c, oh, ow)), output values

        n - batch size
        d - number of input channels
        c - number of output channels
        (ih, iw) - input image shape
        (oh, ow) - output image shape
    """
    # !!! Don't change this function, it's here for your reference only !!!
    assert isinstance(padding, int) and padding >= 0
    assert inputs.ndim == 4 and kernels.ndim == 4
    assert inputs.shape[1] == kernels.shape[1]

    if os.environ.get("USE_FAST_CONVOLVE", False):
        return convolve_pytorch(inputs, kernels, padding)
    else:
        return convolve_numpy(inputs, kernels, padding)


def convolve_numpy(inputs, kernels, padding):
    """
    :param inputs: np.array((n, d, ih, iw)), input values
    :param kernels: np.array((c, d, kh, kw)), convolution kernels
    :param padding: int >= 0, the size of padding, 0 means 'valid'

    :return: np.array((n, c, oh, ow)), output values

        n - batch size
        d - number of input channels
        c - number of output channels
        (ih, iw) - input image shape
        (oh, ow) - output image shape
    """
    # your code here \/
    n, d, ih, iw = inputs.shape
    c, _, kh, kw = kernels.shape

    #паддинг входных данных
    if padding > 0:
        inputs_padded = np.pad(inputs, ((0, 0), (0, 0),
                               (padding, padding), (padding, padding)),
                               mode='constant'
                              )
    else:
        inputs_padded = inputs

    #переворачивает массив вдоль указанных осей, высота и ширина ядра свертки
    # В свертке, в отличие от кросс-корреляции, ядро свертки должно быть отражено по обеим осям
    # (высоте и ширине) перед применением операции. 
    kernels = np.flip(kernels, axis=(-2, -1))

     
    #считаем output image shape
    a = 1 + 2 * padding
    oh = ih - kh + a
    ow = iw - kw + a 
    outputs = np.zeros((n, c, oh, ow))

    for i in range(oh):
        for j in range(ow):
            #подмассив размера ядра свертки
            input_arr = inputs_padded[:, :, i:i+kh, j:j+kw]
            for k in range(c):
                #Выполняем элементное умножение подмассива входных данных и текущего ядра свертки
                #суммируем по каналам, высоте и ширине ядра.
                outputs[:, k, i, j] = np.sum(input_arr * kernels[k, :, :, :], axis=(1, 2, 3))

    return outputs
    # your code here /\


# =============================== 4.1.1 Conv2D ===============================
class Conv2D(Layer):
    def __init__(self, output_channels, kernel_size=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert kernel_size % 2, "Kernel size should be odd"

        self.output_channels = output_channels
        self.kernel_size = kernel_size

        self.kernels, self.kernels_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        output_channels = self.output_channels
        kernel_size = self.kernel_size

        self.kernels, self.kernels_grad = self.add_parameter(
            name="kernels",
            shape=(output_channels, input_channels, kernel_size, kernel_size),
            initializer=he_initializer(input_h * input_w * input_channels),
        )

        self.biases, self.biases_grad = self.add_parameter(
            name="biases",
            shape=(output_channels,),
            initializer=np.zeros,
        )

        self.output_shape = (output_channels,) + self.input_shape[1:]

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, h, w)), input values

        :return: np.array((n, c, h, w)), output values

            n - batch size
            d - number of input channels
            c - number of output channels
            (h, w) - image shape
        """
        # Сохраняем входные данные для использования в backward_impl
        self.forward_inputs = inputs

        # Паддинг для стратегии same
        padding = (self.kernel_size - 1) // 2

        # Выполнение свертки
        result = convolve(inputs, self.kernels, padding=padding) + self.biases.reshape(1, -1, 1, 1)
        return result

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, c, h, w)), dLoss/dOutputs

        :return: np.array((n, d, h, w)), dLoss/dInputs

            n - batch size
            d - number of input channels
            c - number of output channels
            (h, w) - image shape
        """
        # Переворот входных данных и ядер свертки
        flipped_inputs = np.flip(self.forward_inputs, axis=(-2, -1))
        flipped_kernels = np.flip(self.kernels, axis=(-2, -1))

        inv_padding = self.kernel_size - (self.kernel_size - 1) // 2 - 1

        # Вычисление градиентов по ядрам свертки 
        self.kernels_grad = np.transpose((convolve(np.transpose(flipped_inputs, (1, 0, 2, 3)), np.transpose(grad_outputs, (1, 0, 2, 3)), 
                                         (self.kernel_size - 1) // 2)), 
                                         (1, 0, 2, 3)
                                        )
        # Вычисление градиентов по смещениям (biases)
        self.biases_grad = np.sum(grad_outputs, axis=(0, 2, 3)) 
        # Вычисление градиентов по входам
        grad_inputs = convolve(grad_outputs, np.transpose(flipped_kernels, (1, 0, 2, 3)), inv_padding)

        return grad_inputs
        # your code here /\
    
# ============================== 4.1.2 Pooling2D =============================
class Pooling2D(Layer):
    def __init__(self, pool_size=2, pool_mode="max", *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert pool_mode in {"avg", "max"}

        self.pool_size = pool_size
        self.pool_mode = pool_mode
        self.forward_idxs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        channels, input_h, input_w = self.input_shape
        output_h, rem_h = divmod(input_h, self.pool_size)
        output_w, rem_w = divmod(input_w, self.pool_size)
        assert not rem_h, "Input height should be divisible by the pool size"
        assert not rem_w, "Input width should be divisible by the pool size"

        self.output_shape = (channels, output_h, output_w)

    
    
    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, ih, iw)), input values

        :return: np.array((n, d, oh, ow)), output values

            n - batch size
            d - number of channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
        """
        n, d, ih, iw = inputs.shape
        oh, ow = ih // self.pool_size, iw // self.pool_size

        if self.pool_mode == 'max':
            reduce = np.max
        else:
            reduce = np.mean

        # Векторизованное выполнение pooling
        windows = inputs.reshape(n, d, oh, self.pool_size, ow, self.pool_size)
        windows = windows.transpose(0, 1, 2, 4, 3, 5)
        windows = windows.reshape(n, d, oh, ow, self.pool_size * self.pool_size)

        output = reduce(windows, axis=-1)

        # Создание маски для max pooling
        if self.pool_mode == 'max':
            mask = np.zeros_like(inputs, dtype=bool)
            mask_view = inputs.reshape(n, d, oh, self.pool_size, ow, self.pool_size)
            mask_view = mask_view.transpose(0, 1, 2, 4, 3, 5)
            mask_view = mask_view.reshape(n, d, oh, ow, self.pool_size * self.pool_size)

            # найти максимум в каждом
            argmax_at_each_window = mask_view.argmax(axis=-1)
            mask_view[...] = 0
            np.put_along_axis(mask_view, argmax_at_each_window[..., None], 1, axis=-1)

            # reshape
            mask_view = mask_view.reshape(n, d, oh, ow, self.pool_size, self.pool_size)
            mask_view = mask_view.transpose(0, 1, 2, 4, 3, 5)
            mask_view = mask_view.reshape(n, d, ih, iw)

            self.forward_idxs = mask_view.astype(bool)

        return output

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, d, oh, ow)), dLoss/dOutputs

        :return: np.array((n, d, ih, iw)), dLoss/dInputs

            n - batch size
            d - number of channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
        """
        # your code here \/
        #аполняется расширенный тензор, копируя значения из исходного тензора в соответствующие позиции
        def expand(tensor, pool_size):
            expanded_tensor = np.repeat(np.repeat(tensor, pool_size, axis=2), pool_size, axis=3)
            return expanded_tensor

        if self.pool_mode == 'max':
            #Расширить градиенты выходов и умножить их на маску self.forward_idxs,
            #  чтобы распределить градиенты только на те элементы, которые были
            #  выбраны в процессе максимального пула
            grad = expand(grad_outputs, self.pool_size) * self.forward_idxs
        else:
            grad = expand(grad_outputs, self.pool_size) / (self.pool_size**2)
        
        return grad
        # your code here /\


# ============================== 4.1.3 BatchNorm =============================
class BatchNorm(Layer):
    def __init__(self, momentum=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum

        self.running_mean = None
        self.running_var = None

        self.beta, self.beta_grad = None, None
        self.gamma, self.gamma_grad = None, None

        self.forward_inverse_std = None
        self.forward_centered_inputs = None
        self.forward_normalized_inputs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        self.running_mean = np.zeros((input_channels,))
        self.running_var = np.ones((input_channels,))

        self.beta, self.beta_grad = self.add_parameter(
            name="beta",
            shape=(input_channels,),
            initializer=np.zeros,
        )

        self.gamma, self.gamma_grad = self.add_parameter(
            name="gamma",
            shape=(input_channels,),
            initializer=np.ones,
        )

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, h, w)), input values

        :return: np.array((n, d, h, w)), output values

            n - batch size
            d - number of channels
            (h, w) - image shape
        """
        # your code here \/
        if self.is_training:
            # Вычисляем статистики батча
            batch_mean = np.mean(inputs, axis=(0, 2, 3), keepdims=True)
            batch_var = np.var(inputs, axis=(0, 2, 3), keepdims=True)

            #  Нормализуем входные данные
            self.forward_inverse_std = 1 / np.sqrt(batch_var.reshape(-1) + eps) 
            self.forward_centered_inputs = inputs - batch_mean
            self.forward_normalized_inputs = self.forward_centered_inputs * self.forward_inverse_std.reshape(1, -1, 1, 1)
            
            # Обновляем скользящие статистики
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean.reshape(-1)
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var.reshape(-1)

        else:  
            # Используем скользящие статистики
            self.forward_normalized_inputs = ((inputs - self.running_mean.reshape(1, -1, 1, 1))
                                                 / np.sqrt(self.running_var.reshape(1, -1, 1, 1) + eps)
                                             )
        
        output = self.gamma.reshape(1, -1, 1, 1) * self.forward_normalized_inputs + self.beta.reshape(1, -1, 1, 1)
        return output
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, d, h, w)), dLoss/dOutputs

        :return: np.array((n, d, h, w)), dLoss/dInputs

            n - batch size
            d - number of channels
            (h, w) - image shape
        """
        # your code here \/
        # Извлечение размеров
        n, d, h, w = grad_outputs.shape
        total_elements = n * h * w

        # Вычисление градиентов параметров
        self.gamma_grad = np.sum(grad_outputs * self.forward_normalized_inputs, axis=(0, 2, 3))
        self.beta_grad = np.sum(grad_outputs, axis=(0, 2, 3))

        # Преобразование параметров для вычислений
        grad_gamma = self.gamma.reshape(1, d, 1, 1)
        inv_std = self.forward_inverse_std.reshape(1, d, 1, 1)
        inv_norm = self.forward_normalized_inputs

        # Вычисление промежуточных градиентов
        L_derivative = grad_outputs * grad_gamma
        sum_L_derivative = np.sum(L_derivative, axis=(0, 2, 3), keepdims=True)
        sum_L_derivative_norm = np.sum(L_derivative * inv_norm, axis=(0, 2, 3), keepdims=True)

        # Вычисление градиентов по входам
        grad_inputs = (inv_std / total_elements) * (
            total_elements * L_derivative - inv_norm * sum_L_derivative_norm - sum_L_derivative
        )

        return grad_inputs
        # your code here /\


# =============================== 4.1.4 Flatten ==============================
class Flatten(Layer):
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        self.output_shape = (int(np.prod(self.input_shape)),)

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, h, w)), input values

        :return: np.array((n, (d * h * w))), output values

            n - batch size
            d - number of input channels
            (h, w) - image shape
        """
        # your code here \/
        n = inputs.shape[0]
        # -1 : все другие размерности объединяются в одну
        flattened = inputs.reshape(n, -1)
        return flattened
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, (d * h * w))), dLoss/dOutputs

        :return: np.array((n, d, h, w)), dLoss/dInputs

            n - batch size
            d - number of units
            (h, w) - input image shape
        """
        # your code here \/
        n = grad_outputs.shape[0]
        reshaped = grad_outputs.reshape(n, *self.input_shape)
        return reshaped
        # your code here /\


# =============================== 4.1.5 Dropout ==============================
class Dropout(Layer):
    def __init__(self, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.forward_mask = None

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, ...)), input values

        :return: np.array((n, ...)), output values

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        if self.is_training:
            self.forward_mask = np.random.uniform(0, 1, size=inputs.shape) > self.p
            outputs = inputs * self.forward_mask
        else:
            outputs = inputs * (1 - self.p)
        return outputs
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

        :return: np.array((n, ...)), dLoss/dInputs

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        if self.is_training:
            grad_inputs = grad_outputs * self.forward_mask
        else:
            grad_inputs = grad_outputs * (1 - self.p)
        return grad_inputs
        # your code here /\


# ====================== 2.3 Train and Test on CIFAR-10 ======================
def train_cifar10_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    loss = CategoricalCrossentropy()
    optimizer = SGDMomentum(lr=0.01, momentum=0.6)
    model_cifar_101 = Model(loss=loss, optimizer=optimizer)

    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)
    model_cifar_101.add(Conv2D(12, input_shape=(3, 32, 32)))
    model_cifar_101.add(ReLU())       # Затем добавляем функцию активации
    model_cifar_101.add(Pooling2D(pool_mode="max"))  # Слой подвыборки
    # model_cifar_101.add(Dropout(0.25))  # Добавляем Dropout после активации и подвыборки

    model_cifar_101.add(Conv2D(20))
    model_cifar_101.add(BatchNorm())  # Добавляем BatchNorm после сверточного слоя
    model_cifar_101.add(ReLU())       # Затем добавляем функцию активации
    model_cifar_101.add(Pooling2D(pool_mode="max"))  # Слой подвыборки
    # model_cifar_101.add(Dropout(0.2))  # Добавляем Dropout после активации и подвыборки

    model_cifar_101.add(Conv2D(64))
    model_cifar_101.add(BatchNorm())  # Добавляем BatchNorm после сверточного слоя
    model_cifar_101.add(ReLU())       # Затем добавляем функцию активации
    model_cifar_101.add(Pooling2D())  # Слой подвыборки
    model_cifar_101.add(Dropout(0.25))  # Добавляем Dropout после активации и подвыборки

    model_cifar_101.add(Flatten())  # Слой Flatten для преобразования многомерного тензора в одномерный

    model_cifar_101.add(Dense(units=768))
    model_cifar_101.add(ReLU())       # Затем добавляем функцию активации
    model_cifar_101.add(Dropout(0.25))  # Добавляем Dropout после активации

    model_cifar_101.add(Dense(units=128))
    model_cifar_101.add(ReLU())

    model_cifar_101.add(Dense(units=10))
    model_cifar_101.add(Softmax())  # Выходной слой с активацией softmax

    print(model_cifar_101)

    # Обучаем и валидируем модель на предоставленных данных
    model_cifar_101.fit(x_train, y_train, x_valid=x_valid, y_valid=y_valid, epochs=16, batch_size=32)


    # your code here /\
    return model_cifar_101


# ============================================================================
