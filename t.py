import numpy as np 
import cv2 

font = cv2.FONT_HERSHEY_COMPLEX 
img2 = cv2.imread('images/man.png', cv2.IMREAD_COLOR) 


img = cv2.imread('images/man.png', cv2.IMREAD_GRAYSCALE) 

_, threshold = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY) 

contours, _= cv2.findContours(threshold, cv2.RETR_TREE, 
                            cv2.CHAIN_APPROX_SIMPLE) 


# contour_image = np.zeros_like(threshold)
# # cv2.drawContours(img2, contours, -1, (255, 255, 255), 2)




# lines_array = []

# for contour in contours:
#     contour_points = np.squeeze(contour)
#     lines = cv2.HoughLinesP(img, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             cv2.line(img2, (x1, y1), (x2, y2), (255, 0, 0), 2)
#             lines_array.append((x1, y1, x2, y2))

# contours_without_lines = [contour for contour in contours if cv2.contourArea(contour) > 0]


# print(len(contours))
# if len(contours) > 1:
#     contours = contours[1:] 
for cnt in contours[1:] :
    cv2.drawContours(img2, [cnt], 0, (0, 0, 255), 5) 


# Отображаем изображение с контурами и найденными линиями
cv2.imshow("", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()







# cv2.imshow('image2', img2)

# if cv2.waitKey(0) & 0xFF == ord('q'): 
#     cv2.destroyAllWindows()
    
# import matplotlib.pyplot as plt


# plt.plot(x, y)
# plt.show()