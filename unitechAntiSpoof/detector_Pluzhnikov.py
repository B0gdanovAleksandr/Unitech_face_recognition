# -*- coding: utf-8 -*-

import insightface
import numpy as np

from detector import DetectorBase
from PIL import Image
import cv2



class Algorithms(DetectorBase):


    # Функция для предсказания глубины
    def predict_depth(self, image):
        h, w = image.shape[:2]
        # Функция для предобработки изображения
        image = self.depth_transform( Image.fromarray(image) )
        #~ image = self.depth_transform( image )
        processed_image = image.unsqueeze(0)
        with self.no_grad():
            depth = self.depth_model(processed_image)
        depth = depth.squeeze().cpu().numpy()
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        return depth


    def useDepth(self, w_face_depth):
        self.w_face_depth = w_face_depth
        if not w_face_depth:
            return

        # pip install torch torchvision timm
        import torch
        import torchvision.transforms as transforms
        
        self.no_grad = torch.no_grad
        
        # Загрузка модели MiDaS для предсказания глубины
        self.depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True)
        self.depth_model.eval()

        # Трансформации для модели
        self.depth_transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])


    def useFacePosition(self, weight_central, weight_lower):
        self.weight_central = weight_central
        self.weight_lower = weight_lower
        
    def useFaceArea(self, w_face_area):
        self.w_face_area = w_face_area

    def useCropping(self, enable=True):
        self.crop_offset = np.array([0]*4)
        self.cropping = enable


    def cropFrame(self, frame):
        if not self.cropping:
            return frame
        height, width, channels = frame.shape

        crop_width_ratio = 752 / 1920
        crop_height_ratio = 906 / 1080

        left = int(width * 0.25)
        top = 0
        right = int(width * 0.75)
        bottom = int(height * crop_height_ratio)

        self.crop_offset = np.array([left, top]*2)
        
        # Обрезаем изображение
        return frame[top:bottom, left:right]


    def selectFaceOld(self, image_rgb):

        if not self.faces:
            return

        if len(self.faces) == 1:
            self.index = 0
            return self.faces[self.index]

        distances = []
        distances_y = []
        center_x = image_rgb.shape[1] // 2
        for face in self.faces:
            bbox = face['bbox'].astype(int)
            face_center_x = (bbox[0] + bbox[2]) / 2
            face_center_y = (bbox[1] + bbox[3]) / 2
            delta = abs(center_x - face_center_x)
            distances.append(delta)
            distances_y.append(face_center_y)

        min_distance = min(distances)
        candidates = [i for i, d in enumerate(distances) if abs(d - min_distance) / min_distance <= 0.1]

        if len(candidates) > 1:
            self.index = candidates[distances_y.index(min(distances_y[i] for i in candidates))]
        else:
            self.index = distances.index(min_distance)
        return self.faces[self.index]


    def selectFace(self, image):

        if not self.faces:
            return
        if len(self.faces) == 1:
            self.index = 0
            return self.faces[self.index]

        # Расчет баллов по всем методам
        width, height = image.shape[1], image.shape[0]
        methods = []
        if self.weight_central or self.weight_lower:
            methods.append( self.calculate_method1_scores(width, height, self.weight_central, self.weight_lower) )
        if self.w_face_area:
            methods.append( self.calculate_method2_scores(self.w_face_area) )
        if self.w_face_depth:
            self.depth_map = self.predict_depth(image)
            methods.append( self.calculate_method3_scores(self.depth_map, self.w_face_depth) )
        g_scores = map(sum, zip(*methods) )
        scores = tuple(enumerate(g_scores))
        self.index, _ = max(scores, key=lambda x: x[1])

        return self.faces[self.index]
        

    # Функция для расчета баллов по центральной линии и нижней части
    def calculate_method1_scores(self, width, height, weight_central, weight_lower):
        central_line = width / 2
        lower_horizontal = height
        scores = []

        max_distance_central = 0
        max_distance_lower = 0

        # Найти максимальные расстояния для нормализации
        for face in self.faces:
            x_min, y_min, x_max, y_max = face['bbox'].astype(int)
            center_x = (x_min + x_max) / 2
            distance_central = abs(center_x - central_line)
            distance_lower = lower_horizontal - y_max

            max_distance_central = max(max_distance_central, distance_central)
            max_distance_lower = max(max_distance_lower, distance_lower)

        # Расчет баллов
        for face in self.faces:
            x_min, y_min, x_max, y_max = face['bbox'].astype(int)
            center_x = (x_min + x_max) / 2
            distance_central = abs(center_x - central_line)
            distance_lower = lower_horizontal - y_max

            score_central = 1 - (distance_central / max_distance_central) if max_distance_central > 0 else 1
            score_lower = 1 - (distance_lower / max_distance_lower) if max_distance_lower > 0 else 1

            total_score = score_central * weight_central + score_lower * weight_lower
            scores.append(total_score)

        return scores


    # Функция для расчета баллов по максимальной площади
    def calculate_method2_scores(self, w_face_area):
        scores = []
        max_area = 0

        # Найти максимальную площадь для нормализации
        for face in self.faces:
            x_min, y_min, x_max, y_max = face['bbox'].astype(int)
            area = (x_max - x_min) * (y_max - y_min)
            max_area = max(max_area, area)

        # Расчет баллов
        for face in self.faces:
            x_min, y_min, x_max, y_max = face['bbox'].astype(int)
            area = (x_max - x_min) * (y_max - y_min)
            score = area / max_area if max_area > 0 else 1
            scores.append(score * w_face_area)

        return scores


    # Функция для расчета баллов по глубине
    def calculate_method3_scores(self, depth_map, w_face_depth):
        scores = []
        max_depth = 0

        # Найти максимальную глубину для нормализации
        for face in self.faces:
            x_min, y_min, x_max, y_max = face['bbox'].astype(int)
            face_depth_region = depth_map[y_min:y_max, x_min:x_max]
            if np.any(np.isnan(face_depth_region)) or np.any(face_depth_region == 0):
                face_depth = np.nan
            else:
                face_depth = np.mean(face_depth_region)

            max_depth = max(max_depth, face_depth) if not np.isnan(face_depth) else max_depth

        # Расчет баллов
        for face in self.faces:
            x_min, y_min, x_max, y_max = face['bbox'].astype(int)
            face_depth_region = depth_map[y_min:y_max, x_min:x_max]
            if np.any(np.isnan(face_depth_region)) or np.any(face_depth_region == 0):
                face_depth = np.nan
            else:
                face_depth = np.mean(face_depth_region)

            score = face_depth / max_depth * w_face_depth if max_depth > 0 and not np.isnan(face_depth) else 0
            scores.append(score)

        return scores




# Класс для детекции лиц, наследующийся от базового класса DetectorBase
class Recognizer(Algorithms):

    def __init__(self):
        super().__init__()
        """
        Конструктор класса FaceDetector.
        Инициализирует объект FaceAnalysis из библиотеки insightface для обработки лиц.
        """

        self.faces = []
        self.index = 0
        
        # Накопить `embedding_count` эмеддингов для нового пользователя
        #~ self.embedding_count = 10
        self.embeddings = []
        
        # Создание объекта FaceAnalysis с использованием модели "buffalo_sc"
        # Разрешены модули: 'detection' для обнаружения лиц, 'recognition' для распознавания лиц,
        # и 'landmark_3d_68' для определения 3D-меток на лице
        self.face_app = insightface.app.FaceAnalysis(
            name="buffalo_sc",
            allowed_modules=('detection', 'recognition')
        )
        # Подготовка модели, установка порога детекции и размера входного изображения
        self.face_app.prepare(ctx_id=0, det_thresh=0.7, det_size=(256, 256))

        self.useCropping(False)
        self.useFacePosition(0.4, 0.2)
        self.useFaceArea(4)
        self.useDepth(1)

        # Порог и тип сравнения эмбеддингов лиц
        self.useDistanceMetric('cosine', 0.7)
    

    def scaningNewUser(self, rgb):
        rgb = self.cropFrame(rgb)
        
        # Обнаружение лиц на изображении
        self.faces = self.face_app.get(rgb)
        
        face = self.selectFace(rgb)
        if not face:
            return

        # Извлечение эмбеддинга лица
        self.embeddings = [ face.embedding ]
        return True


    def recognizingUser(self, rgb, metadata):

        rgb = self.cropFrame(rgb)
        
        # Обнаружение лиц на изображении
        self.faces = self.face_app.get(rgb)

        # Если лица не найдены, возвращаем None
        face = self.selectFace(rgb)
        if not face:
            return

        # Поиск совпадений эмбеддингов в базе данных
        return self.searchGuidInDataframe(face.embedding)
            


# Класс для детекции лиц, наследующийся от базового класса DetectorBase
class FaceDetector(Recognizer):

    def __init__(self):
        super().__init__()
        """
        Конструктор класса FaceDetector.
        Инициализирует объект FaceAnalysis из библиотеки insightface для обработки лиц.
        """
        self.useCropping()

 
    def selectFace(self, rgb):
        return self.selectFaceOld(rgb)


