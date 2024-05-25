import cv2
import numpy as np

def find_and_remove_lines(contour_points, threshold_distance=10):
    # Преобразование набора точек в массив numpy
    points = np.array(contour_points)

    # Создание пустого изображения
    blank_image = np.zeros((500, 500, 3), np.uint8)

    # Создание изображения с точками из набора
    for point in points:
        cv2.circle(blank_image, tuple(point), 3, (255, 255, 255), -1)

    # Преобразование изображения в градации серого
    gray = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)

    # Применение алгоритма Кэнни для детектирования границ
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Обнаружение прямых линий на изображении
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Нахождение расстояния между двумя точками прямой и всех точек набора
            distances = np.linalg.norm(points - np.array([[x1, y1]]), axis=1) + np.linalg.norm(points - np.array([[x2, y2]]), axis=1)
            # Поиск точек, расстояние до которых меньше порогового значения
            points_to_remove = np.where(distances < threshold_distance)[0]
            # Удаление найденных точек из набора
            points = np.delete(points, points_to_remove, axis=0)

    return points

# Пример использования
# Замените этот массив на ваш набор точек
contour_points = [(100, 100), (200, 100), (300, 200), (400, 200), (500, 300), (600, 300)]

# Вызов функции
remaining_points = find_and_remove_lines(contour_points)

print("Оставшиеся точки после удаления прямых линий:")
print(remaining_points)
