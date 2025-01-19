# -*- coding: utf-8 -*-

import logging
import numpy as np

import detector
import detect_embedding as dem
import detector_Pluzhnikov


# назначить имя и получить логер
logger = logging.getLogger(__name__)


class EmbeddingCreator(detector_Pluzhnikov.Algorithms):


    def __init__(self, detector='dlib', embedding='Facenet512'):
        super().__init__()

        self.useCropping()
        self.faces = []
        self.index = 0
        
        # Накопить `embedding_count` эмеддингов для нового пользователя
        self.embedding_count = 10
        self.embeddings = []

        # выбор фреймворка
        if 'buffalo' in detector:
            if 'buffalo' in embedding:
                self.det_framework = dem.Insightface(name=embedding, allowed_modules=('detection', 'recognition'))
            else:
                self.det_framework = dem.Insightface(name=detector, allowed_modules=('detection'))
        elif 'haarcascade' in detector:
            self.det_framework = dem.HaarCascade()
        elif 'dlib' in detector:    # Deep_Face(detector_backend='dlib'     тормозит
            self.det_framework = dem.Dlib()
            logger.info('External dlib loaded.')
        else:
            self.det_framework = dem.Deep_Face(detector_backend=detector)
        if 'buffalo' in embedding:
            if 'buffalo' in detector:
                self.framework = self.det_framework
            else:
                self.framework = dem.Insightface(name=embedding, allowed_modules=('detection', 'recognition'))
        else:
            self.framework = dem.Deep_Face(model_name=embedding)

        self.useCropping(False)
        self.useFacePosition(0.4, 0.2)
        self.useFaceArea(4)
        self.useDepth(0)


    def face_select(self, rgb):
        #~ x1,y1, x2,y2 = faces[0]['bbox'].astype(int)
        #~ eyes = self.det_framework.predict(rgb[y1:y2, x1:x2], np.array( [x1, y1]*2 ))      # HaarCascade
        #~ faces.extend({'bbox': e} for e in eyes)

        #~ points = self.det_framework.predict(rgb, faces[0]['bbox'])         # Dlib
        #~ for i, (x,y) in enumerate(points):
            #~ cv2.circle(rgb, (x, y), 2, 255, 1)
            #~ cv2.putText(rgb, str(i), (x + 4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255))
        self.face_selected = True

    def detector(self, frame):
        #~ if self.det_framework is None:
            #~ raise TypeError('В InsightFace нет отдельного детектора!')
        return self.det_framework.detector(frame)
    
    def embedding(self, frame, bbox):
        #~ if self.det_framework is None:
            #~ raise TypeError('Для InsightFace используйте `metrics_from(frame)`!')
        x1,y1, x2,y2 = bbox.astype(int)
        face_image = frame[y1:y2, x1:x2]
        return self.framework.embedding(face_image)
    
    def metrics_from(self, frame):
        if self.det_framework is self.framework:
            return self.framework.detector(frame)
        faces = self.det_framework.detector(frame)
        for face in faces:
            face['embedding'] = self.embedding(frame, face['bbox'])
        return faces

    def scaningNewUser(self, rgb):

        rgb = self.cropFrame(rgb)

        self.faces = self.detector(rgb)
        
        face = self.selectFace(rgb)
        if not face:
            return

        embedding = self.embedding(rgb, face['bbox'])
        if len(embedding):
            self.embeddings.append( embedding )
            if len(self.embeddings) == self.embedding_count:
                return True

    def recognizingUser(self, rgb, metadata):

        rgb = self.cropFrame(rgb)

        self.faces = self.detector(rgb)
        
        face = self.selectFace(rgb)
        if not face:
            return

        embedding = self.embedding(rgb, face['bbox'])
        if len(embedding):
            # Поиск совпадений эмбеддингов в базе данных
            return self.searchGuidInDataframe(embedding)
            
