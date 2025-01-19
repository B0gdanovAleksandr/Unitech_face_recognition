import cv2
import onnxruntime as ort
import numpy as np
import os

# onnx model
class AntiSpoof:
    def __init__(self, weights: str):
        """Инициализация объекта класса AntiSpoof

        Args:
            weights (str, Optional): Путь до весов модели. Defaults to None.
            model_img_size (int, optional): Размер большей стороны изображения, которое будет принимать модель.
            Defaults to 128.
        """
        #super().__init__()
        self.weights = weights if isinstance(weights, str) else None
        self.model_img_size = 128
        self.ort_session, self.input_name = self._init_session_(self.weights)

    def _init_session_(self, onnx_model_path: str):
        """Инициализация модели

        Args:
            onnx_model_path (str): self.weights

        Returns:
            object, str: объект для инференса модели, имя входного тензора
        """
        
        ort_session = None
        input_name = None
        if os.path.isfile(onnx_model_path):
            try:
                #Мало ли c GPU терминал появится :)
                ort_session = ort.InferenceSession(onnx_model_path, 
                                                   providers=['CUDAExecutionProvider'])
            except:
                #Наш вариант
                ort_session = ort.InferenceSession(onnx_model_path, 
                                                   providers=['CPUExecutionProvider']) 
            input_name = ort_session.get_inputs()[0].name
        return ort_session, input_name

    def preprocessing(self, img): 
        """Предобработка входного изображения в модель

        Args:
            img (nd.array RGB!!!): изображение, которое обязательно в формате numpy
            массива и с цветовым пространством RGB 

        Returns:
            _type_: _description_
        """
        #ПРИ ПОДАЧЕ ДОЛЖНО БЫТЬ: img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        new_size = self.model_img_size
        old_size = img.shape[:2] # формат (height, width), т.е. nd.array формат
        #получение нового размера изображения для подачи в модель
        ratio = float(new_size)/max(old_size)
        scaled_shape = tuple([int(x*ratio) for x in old_size])

        img = cv2.resize(img, (scaled_shape[1], scaled_shape[0]))

        delta_w = new_size - scaled_shape[1]
        delta_h = new_size - scaled_shape[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)

        img = cv2.copyMakeBorder(img, top, bottom, left, right, 
                                 cv2.BORDER_CONSTANT, value=[0, 0, 0])
        #для torch (H, W, C) -> (C, H, W)
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    def postprocessing(self, prediction) -> list:
        """Метод постобработки предсказаний модели
        Применяется ф-ция активации softmax для выходных
        значений классификатора 

        Args:
            prediction (nd.array): выходные значения классификатора

        Returns:
            list: список вероятносетй принадлежности изображения
            к определённому классу (настоящий, ненастоящий, ненастоящий)
        """
        softmax = lambda x: np.exp(x)/np.sum(np.exp(x))
        pred = softmax(prediction)
        return pred
        #return np.argmax(pred)

    def __call__(self, imgs : list):# -> bool|list:
        """Метод вызова объекта класса

        Args:
            imgs (list): список изображений, если одно, то задаётся [img]

        Returns:
            bool: False, в случае если модель не сработает
            list: список предсказаний модели для каждого поданного изображения,
            где элемент с индексом [0][0] - вероятность, что на первом изображении
            настоящий человек и т.п.
        """
        if not self.ort_session:
            return False

        preds = []
        for img in imgs:
            onnx_result = self.ort_session.run([],
                {self.input_name: self.preprocessing(img)})
            pred = onnx_result[0]
            pred = self.postprocessing(pred)
            preds.append(pred)
        return preds
    
    @staticmethod
    def increased_crop(img, bbox: tuple, bbox_inc: float = 1.5, marking='yolo'):
        """Статический метод для получения обрезанного лица с увеличенной областью

        Args:
            img (nd.array): исходный кадр/изображение которое хотим проверить на живость  
            bbox (tuple): координаты ограничивающей рамки
            bbox_inc (float, optional): коэффициет увеличения. Defaults to 1.5.
            marking (str, optional): формат разметки, доступные 'coco', 'voc', 'yolo'. Defaults to 'yolo'.
        """
        real_h, real_w = img.shape[:2]
        x, y, w, h = bbox
        if marking == 'yolo' or marking == 'coco':
                pass
        elif marking == 'voc':
                w, h = w - x, h - y
        else:
                raise ValueError("Недопустимый формат разметки")
                
        l = max(w, h)
        xc, yc = x + w/2, y + h/2
        x, y = int(xc - l*bbox_inc/2), int(yc - l*bbox_inc/2)
        x1 = 0 if x < 0 else x 
        y1 = 0 if y < 0 else y
        x2 = real_w if x + l*bbox_inc > real_w else x + int(l*bbox_inc)
        y2 = real_h if y + l*bbox_inc > real_h else y + int(l*bbox_inc)

        img = img[y1:y2,x1:x2,:]
        img = cv2.copyMakeBorder(img, 
                                    y1-y, int(l*bbox_inc-y2+y), 
                                    x1-x, int(l*bbox_inc)-x2+x, 
                                    cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return img
    
    def __str__(self) -> str:
        return f"Модель для определения живости лица. Weights: {self.weights}"