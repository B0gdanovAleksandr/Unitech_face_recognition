# -*- coding: utf-8 -*-

	
import io
import sys
import pathlib
import cv2	

import numpy as np

from PIL import Image


def g_thin_out_frames(g_frames, step=1):
    """
    Отбрасывает все кадры кроме каждого `step`-го кадра.
    """
    # Счетчик кадров
    frame_count = 0
        
    for rgb_frame, metadata in g_frames:

        # Берем каждый step-ый кадр
        frame_count %= step
        
        if not frame_count:
            yield rgb_frame, metadata

        frame_count += 1


def g_from_directory(dataset_path, file_mask='*.*'):
    """
    Создает поток кадров из картинок в каталоге.
    """
    images_path = pathlib.Path(dataset_path)

    for file in sorted(images_path.glob(file_mask)):
        with Image.open(file) as image:
            yield np.array(image), file.name

def g_from_one_file(dataset_path):
    """
    Создает поток кадров из одного файла.
    """
    file = pathlib.Path(dataset_path)

    with Image.open(file) as image:
        yield np.array(image), file.name


def g_from_cv_capture(file_path):
    """
    Создает поток кадров из устройства захвата или видео файла.
    """
    try:
        cap = cv2.VideoCapture(file_path) # file_path can be number if you need to use cam device (-1 for default)
        if isinstance(file_path, int) and not cap.isOpened():
            raise PermissionError("Cannot open camera!")
        
        frame_count = 0
        
        while True:
            
            # Читаем кадр из видео
            success, frame = cap.read()
            if not success:
                break

            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), frame_count
            
            frame_count += 1
    finally:
        cap.release()


