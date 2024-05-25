from tkinter import filedialog
import cv2
import numpy as np
import svgwrite

def convert_to_svg(input_image_path, output_svg_path):
    # Загрузка изображения
    image = cv2.imread(input_image_path)
    
    # Преобразование в градации серого
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Применение гауссовского размытия для сглаживания изображения
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Применение адаптивной бинаризации для получения бинарного изображения
    binary_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)
    
    # Применение морфологических операций для улучшения качества бинарного изображения
    kernel = np.ones((3,3), np.uint8)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Нахождение контуров
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Создание SVG объекта
    svg_document = svgwrite.Drawing(output_svg_path, profile='tiny')
    
    # Добавление контуров в SVG
    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = np.squeeze(approx).tolist()
        svg_document.add(svg_document.polygon(points, fill='none', stroke='black'))
    
    # Сохранение SVG файла
    svg_document.save()
    
    print("Конвертация завершена.")

# Пример использования
file_path = filedialog.askopenfilename()
if file_path:
    convert_to_svg(file_path, "output_image.svg")
