from collections import defaultdict
import json
import os
import tkinter as tk
from tkinter import filedialog, ttk
import cv2
from PIL import Image, ImageTk
import numpy as np


class ContourApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Contour Selection App")
        
        self.frame = tk.Frame(self.root)
        self.frame.pack(fill=tk.BOTH, expand=True)

        self.image_label = tk.Label(self.frame)
        self.image_label.pack()

        self.canvas = tk.Canvas(self.frame)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.threshold_label = tk.Label(self.frame, text="Порог:")
        self.threshold_label.pack(side=tk.LEFT, padx=5, pady=5)
        self.threshold_scale = tk.Scale(self.frame, from_=0, to=255, orient=tk.HORIZONTAL)
        self.threshold_scale.set(127)
        self.threshold_scale.pack(side=tk.LEFT, padx=5, pady=5)
        self.threshold_scale.bind("<ButtonRelease>", self.update_threshold)


        self.scrollbar = ttk.Scrollbar(self.frame, orient=tk.VERTICAL, command=self.canvas.
                yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind_all("<MouseWheel>", self.on_mousewheel)
        
        self.contour_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.contour_frame, anchor="nw")

        # self.canvas["yscrollcommand"]=self.scrollbar.set

        self.load_button = tk.Button(self.root, text="Загрузить изображение", command=self.load_image)
        self.load_button.pack()
        
        self.save_button = tk.Button(self.root, text="Сохранить контуры", command=self.save_contours)
        self.save_button.pack(side=tk.RIGHT, padx=5, pady=5)


        self.select_all_button = tk.Button(self.root, text="Выделить все", command=self.select_all_contours)
        self.select_all_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.unselect_all_button = tk.Button(self.root, text="Убрать все", command=self.unselect_all_contours)
        self.unselect_all_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.merge_contours_button = tk.Button(self.root, text="Объединить контуры", command=self.merge_contours)

        self.merge_contours_button = tk.Button(self.root, text="Усреднить по Y", command=self.average)
        self.merge_contours_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.check_button = tk.Button(self.root, text="Отрисовать", command=lambda : os.system('draw.py'))
        self.check_button.pack(side=tk.LEFT, padx=5, pady=5)


        self.contour_checkboxes = []
        self.avg_list = []
        self.selected_contours = []

    def update_threshold(self, event):
        try:
            self.load_image(self.path)
        except:
            pass
    
    def on_mousewheel(self, event):
        self.canvas.yview_scroll(-1 * int(event.delta / 120), "units")
        
    def save_contours(self):
        points = []
        for i, contour in enumerate(self.contours):
            if self.contour_checkboxes[i].instate(['selected']):
                if i in self.avg_list:
                    d = defaultdict(list)
                    for x,y in contour.reshape(-1,2):
                        d[x].append(y)
                    for x in sorted(d):
                        points.append((x, np.array(d[x]).mean()))
                else:
                    for x,y in contour.reshape(-1,2):
                        points.append((x, y))
                points.append(('end',''))                
        with open("output.txt", "w") as file:
            for point in points:
                file.write(f"{point[0]},{point[1]}\n")
        print("Контуры сохранены в файл output.txt")

    def average(self):
        # averaged_contours = []
        # remaining_contours = []
        # for i, contour in enumerate(self.contours):
        #     if self.contour_checkboxes[i].instate(['selected']):
                # d = defaultdict(list)
                # for x,y in contour.reshape(-1,2):
                #     d[x].append(y)
                # t = []
                # for x in sorted(d):
                #     t.append([x, np.array(d[x]).mean()])
        #         # print(np.array(t).reshape((-1, 1, 2)))
        #         averaged_contours.append(np.array(t, np.float32).reshape((-1, 1, 2)))
        #     else:
        #         remaining_contours.append(contour)
        # self.contours = remaining_contours + averaged_contours
        # self.update_image()
        # self.create_contour_checkboxes()
        # print("Контуры усреднены")
        for i, contour in enumerate(self.contours):
            if self.contour_checkboxes[i].instate(['selected']):
                self.avg_list.append(i)
            
    def merge_contours(self):
        merged_contour_points = []
        remaining_contours = []
        for i, contour in enumerate(self.contours): 
            if self.contour_checkboxes[i].instate(['selected']):
                contour_points = contour.reshape(-1, 2)
                merged_contour_points.extend(contour_points.tolist())
            else:
                remaining_contours.append(contour)
        self.contours = remaining_contours + [np.array(merged_contour_points)]
        self.update_image()
        self.create_contour_checkboxes()
        print("Контуры объединены")


    def load_image(self, path=''):
        if path:
            file_path=path
        else:
            file_path = filedialog.askopenfilename()
        self.path = file_path
        threshold_value = self.threshold_scale.get()
        if file_path:
            self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            self.original_image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            _, threshold = cv2.threshold(self.image, threshold_value, 255, cv2.THRESH_BINARY) 
            self.contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # self.show_image()
            self.create_contour_checkboxes()
            self.update_image()

    # def show_image(self):
    #     self.selected_contours.clear()
    #     contour_img = self.original_image.copy()
    #     for i, contour in enumerate(self.contours):
    #         cv2.drawContours(contour_img, [contour], -1, (0, 255, 0), 2)
    #         self.selected_contours.append(True)
    #     self.photo = ImageTk.PhotoImage(image=Image.fromarray(contour_img))
    #     self.image_label.config(image=self.photo)

    def create_contour_checkboxes(self):
        for checkbox in self.contour_checkboxes:
            checkbox.destroy()
        self.contour_checkboxes.clear()

        for i in range(len(self.contours)):
            var = tk.BooleanVar(value=True)
            checkbox = ttk.Checkbutton(self.contour_frame, text=f"Контур {i}", variable=var, command=self.update_image)
            checkbox.pack(anchor=tk.W)
            self.contour_checkboxes.append(checkbox)

    def update_image(self):
        contour_img = self.original_image.copy()
        for i, contour in enumerate(self.contours):
            if self.contour_checkboxes[i].instate(['selected']):
                cv2.drawContours(contour_img, [contour], -1, (0, 255, 0), 2)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(contour_img).resize(size=(512,512)))
        self.image_label.config(image=self.photo)
    
    def select_all_contours(self):
        for checkbox in self.contour_checkboxes:
            checkbox.state(['selected'])
        self.update_image()

    def unselect_all_contours(self):
        for checkbox in self.contour_checkboxes:
            checkbox.state(['!selected'])
        self.update_image()
if __name__ == "__main__":
    root = tk.Tk()
    
    app = ContourApp(root)
    root.mainloop()
