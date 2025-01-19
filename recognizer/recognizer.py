# -*- coding: utf-8 -*-

import logging
from pathlib import Path

import cv2
from tools import view

from unitech.Reznik import AntiSpoofSystem as RecognizerClass
#~ from unitech.detector_Pluzhnikov import Recognizer as RecognizerClass
#~ from unitech.detector_stepanov import EmbeddingCreator as RecognizerClass

class Recognizer(RecognizerClass):

    
    def __init__(self, database_path, spoof_model_path=None):
        if spoof_model_path is None:
            super().__init__()
        else:
            super().__init__(spoof_model_path)
        self.useThreading(False)
        self.setDB(database_path)

        # Получать видео с ВЭБ камеры
        self.setAdressCamera(0)
        
        # Отключить защиту от спуфинга
        #~ self.useAlivePerson(0)


    def faceRecognitionAll(self, not_test=True):
        self.not_test = not_test
        return super().faceRecognitionAll()


    def scanNewUser(self):
        """
        Метод для добавления информации о пользователе.

        :return: Эмбеддинги лица, извлеченные из изображения.
        """
        self.embeddings = []
        with view.Viewer(resize_factor=2) as viewer:

            # Поток кадров изображения и метаданных.
            for rgb, metadata in self.frame_stream():
                if self.scaningNewUser(rgb):
                    break
                color = (255,255,0) if self.spoof_detected else (0,255,0)
                viewer.draw_bboxes(rgb, self.faces, [self.index, color], self.crop_offset)
                if not viewer.imshow(rgb):
                    break
        return self.embeddings


    def recognizeUser(self):
        """
        Метод для получения информации о пользователе по изображению.

        :return: Идентификатор пользователя, если лицо найдено, иначе None.
        """
        guid = ''
        with view.Viewer(resize_factor=2) as viewer:

            # Поток кадров изображения и метаданных.
            for rgb, metadata in self.frame_stream():
                guid = self.recognizingUser(rgb, metadata)
                if guid:
                    cv2.putText(rgb, guid, (0,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
                color = (255,255,0) if self.spoof_detected else (0,255,0)
                viewer.draw_bboxes(rgb, self.faces, [self.index, color], self.crop_offset)
                if not viewer.imshow(rgb):
                    return guid
                if guid and self.not_test :
                    return guid
        return guid


    def viewAllUserInBd(self):
        print(self.dataframe['guid'])
        

if __name__ == "__main__":
    
    recognizer = Recognizer('../report/recognizer.db')
    
    not_found = '\033[91mПользователь не найден!\033[0m'
    guid_found = '\033[92mGUID пользователя: %s\033[0m'
    not_alive = '\033[93mЛицо не является живым человеком!\033[0m'

    while True:

        print("\n\nВыберите режим работы:")
        print("[0] Тест распознавания")
        print("[1] Распознавание")
        print("[2] Регистрация пользователя")
        print("[3] Просмотр всех пользователей в БД")
        print("[4] Удаление по id")
        print("[5] Удаление всех пользователей из БД")
        print("Нажмите 'x' для выхода.")

        char = input()

        if char == "x":
            break
        if char == "0":
            print("ТЕСТ РАСПОЗНАВАНИЯ\n")
            guid = recognizer.faceRecognitionAll(False)
            print(not_alive if recognizer.spoof_detected else guid_found % guid if guid else not_found )
            continue
        if char == "1":
            print("РЕЖИМ РАСПОЗНАВАНИЯ\n")
            guid = recognizer.faceRecognitionAll()
            print(not_alive if recognizer.spoof_detected else guid_found % guid if guid else not_found )
            continue
        if char == "2":
            print("РЕЖИМ РЕГИСТРАЦИИ\n")
            guid = recognizer.beginScan()
            print(not_alive if recognizer.spoof_detected else guid_found % guid if guid else not_found )
            continue
        if char == "3":
            print("РЕЖИМ ПРОСМОТРА ВСЕХ ПОЛЬЗОВАТЕЛЕЙ В БД\n")
            recognizer.viewAllUserInBd()
            continue
        if char == "4":
            print("РЕЖИМ УДАЛЕНИЯ ПОЛЬЗОВАТЕЛЯ ПО ID\n")
            print("Введите ID пользователя: ")
            guid = input()
            recognizer.deleteDataById(guid)
            continue
        if char == "5":
            print("РЕЖИМ УДАЛЕНИЯ ВСЕХ ПОЛЬЗОВАТЕЛЕЙ ИЗ БД\n")
            recognizer.deleteAllFromData()
            print("Все пользователи удалены ")
            continue
        print("Неверный выбор. Попробуйте снова.")
    