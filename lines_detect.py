# import cv2
# import numpy as np

# # Загрузка изображения и преобразование в оттенки серого
# image = cv2.imread('images/eight.jpg')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Применение детектора границ Кэнни
# edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# # Преобразование Хафа для обнаружения линий
# lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)

# # Нарисовать обнаруженные линии на изображении
# if lines is not None:
#     i = 0
#     for line in lines:
#         print(line)
#         x1, y1, x2, y2 = line[0]
#         print(f'line {i}: ({x1}, {y1}), ({x2}, {y2})')
#         i += 1
#         cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# # Отображение результата
# cv2.imshow('Detected Lines', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import cv2
import numpy as np

# Загрузка изображения и преобразование в оттенки серого
image = cv2.imread('images/man.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Применение детектора границ Кэнни
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Преобразование Хафа для обнаружения линий
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)

# Фильтрация дубликатов
unique_lines = []
if lines is not None:
    i = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Рассчитываем параметры линии: угол и длину
        angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        # Проверяем, есть ли уже такая линия в списке уникальных линий
        is_unique = True
        for unique_line in unique_lines:
            distance = np.linalg.norm(np.array([x1, y1]) - np.array([unique_line[1][0], unique_line[1][1]]))
            # Проверяем, являются ли линии параллельными
            if np.abs(angle - unique_line[0]) < 10 and distance < 10:
                is_unique = False
                break
        if is_unique:
            unique_lines.append((angle, (x1, y1), (x2, y2)))
            print(f'line {i}: ({x1}, {y1}), ({x2}, {y2})')
            i += 1
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Отображение результата
cv2.imshow('Detected Lines', image)
cv2.waitKey(0)
cv2.destroyAllWindows()