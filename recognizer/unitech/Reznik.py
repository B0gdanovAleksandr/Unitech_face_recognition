from unitech.detector_Pluzhnikov import Recognizer
from unitech.AntiSpoof.FaceAntiSpoofing import AntiSpoof


class AntiSpoofSystem(Recognizer):
    
    def __init__(self, spoof_model_path='unitech/saved_models/AntiSpoofing_bin_1.5_128.onnx') -> None:
        """Конструктор
        """
        super().__init__()
        self.spoof_detected = False
        self.antispoof_model = AntiSpoof(spoof_model_path)
        self.useAlivePerson()
        
    def useAlivePerson(self, spoof_threshold=0.6, frames_threshold=0.9, frame_count=10):
        """инициализация порога или выключатель при `threshold=0`
        Args:
            spoof_threshold (int | float): порог [0;1]
            инициализация модели и порога
            frames_threshold (int | float): порог [0;1]
            пороговое отношение живых кадров к `frame_count`
            frame_count (int ): 
            количество кадров взятых для анализа
        """
        self.threshold = spoof_threshold
        self.frames_threshold = frames_threshold
        self.frame_count = frame_count

    def frame_stream(self):
        self.scores = [0] * self.frame_count
        yield from super().frame_stream()
    
    def selectFace(self, image):
        face = super().selectFace(image)
        if face is None:
            return
        if not self.threshold:
            return face
        self.checkAlivePerson(image, face['bbox'])
        if self.setAlivePerson():
            return face
    
    def setAlivePerson(self):
        """вынос вердикта (настоящий/фальшивый)

        Returns:
            bool: окончательный вердикт
        """
        # super().setAlivePerson(seconds)
        #~ valid_score = seconds*fps*self.threshold
        if sum(self.scores) / self.frame_count > self.frames_threshold:
            self.spoof_detected = False
            return True
        self.spoof_detected = True
    
    def checkAlivePerson(self, frame, bbox: tuple):
        """работа модели (анализ каждого кадра на живость)

        Args:
            frame (nd.array): кадр видеопотока
            bbox (tuple): ограничивающая рамка лица
        """
        frame = AntiSpoof.increased_crop(frame, bbox, 2, 'voc')
        scores = self.antispoof_model([frame])[0]
        if scores[0][0] > self.threshold:
            self.scores.append(1)
        else:
            self.scores.append(0)
        self.scores.pop(0)
