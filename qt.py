import operator
import random
import numpy as np
from deap import algorithms, base, creator, gp, tools
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QFileDialog, QComboBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import sympy as sp
from threading import Thread
import sys

def protectedDiv(left, right):
    return left / right if right else 1

def sqrt(x):
    return np.sqrt(np.abs(x))

def simplify_formula(ind):
    # Преобразуем формулу в строку
    locals = {
    'sub': lambda x, y : x - y,
    'div': lambda x, y : x/y,
    'mul': lambda x, y : x*y,
    'add': lambda x, y : x + y,
    'neg': lambda x : -x,
    }

    # ind = 'div(add(div(x, mul(y, y)), 1), mul(x, y))'
    # print(f'original: {ind}')
    expr = sp.sympify(str(ind) , locals=locals)
    return expr


# Создание класса FitnessMin для минимизации ошибки
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# Определение функций и терминалов для генетического программирования
pset = gp.PrimitiveSet("MAIN", arity=1)
pset.addPrimitive(operator.add, arity=2)
pset.addPrimitive(operator.sub, arity=2)
pset.addPrimitive(operator.mul, arity=2)
pset.addPrimitive(protectedDiv, arity=2)
pset.addPrimitive(sqrt, arity=1)
pset.addPrimitive(np.sin, arity=1)
pset.addPrimitive(np.cos, arity=1)
pset.addPrimitive(np.tan, arity=1)
pset.addEphemeralConstant("rand", lambda: random.uniform(-1, 1))
pset.renameArguments(ARG0='x')

# Определение инструментов для генетического программирования
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# Определение функции оценки (evaluation function)
def evaluate(individual, points):
    func = toolbox.compile(expr=individual)
    sqerrors = ((func(x) - y)**2 for x, y in points)
    return np.sum(sqerrors),

# Создание GUI
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Symbolic Regression App")
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)


        # Создание кнопки для выбора файла с данными
        self.load_button = QPushButton("Загрузить данные")
        self.load_button.clicked.connect(self.load_data)
        layout.addWidget(self.load_button)

        self.invert_button = QPushButton("Обратить y")
        self.invert_button.clicked.connect(self.invert_y)
        layout.addWidget(self.invert_button)

        self.start_button = QPushButton("Начать")
        self.start_button.clicked.connect(self.run_evolution)
        layout.addWidget(self.start_button)

        self.prediction_combo = QComboBox()
        self.prediction_combo.currentIndexChanged.connect(self.update_prediction)
        layout.addWidget(self.prediction_combo)

        # Создание виджета для отображения графика
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.x_points = None
        self.y_points = None
        self.predictions = []


    def invert_y(self):
        self.y_points = np.array(self.y_points)
        self.y_points = max(self.y_points) - self.y_points
        self.draw_data()
    
    def draw_data(self):
        # Отображение загруженных данных
        plt.cla()
        plt.scatter(self.x_points, self.y_points, color='red', label="Точки данных")
        plt.axis('equal')
        plt.legend()
        self.canvas.draw()
    
    def load_data(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Data File", "", "Text Files (*.txt)", options=options)
        if file_name:
            self.x_points, self.y_points = [], []
            with open(file_name, "r") as f:
                for line in f:
                    if line.strip() == "end,":
                        break
                    x, y = map(float, line.split(','))
                    self.x_points.append(x)
                    self.y_points.append(y)
            self.draw_data()
            
    def update_prediction(self, index):
        if self.predictions:
            plt.cla()
            
            y_predicted = [self.predictions[index][0](xi) for xi in self.x_points]
            plt.scatter(self.x_points, self.y_points, color='red', label="Actual")
            plt.plot(self.x_points, y_predicted, label=f"Выражение {index+1}")
            plt.axis('equal')
            plt.legend()
            plt.title(f"Формула: {self.predictions[index][1]}\nОшибка: {self.predictions[index][2]:.4f}")
            self.canvas.draw()

    def run_evolution(self):
        if self.x_points is None or self.y_points is None:
            print("Data not loaded.")
            return

        # Определение функции оценки с загруженными данными
        toolbox.register("evaluate", evaluate, points=list(zip(self.x_points, self.y_points)))

        # Создание экземпляра Population и остальных параметров генетического алгоритма
        population = toolbox.population(n=300)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("mutate", gp.mutNodeReplacement, pset=pset)
        toolbox.register("select", tools.selTournament, tournsize=3)

        # Запуск генетического алгоритма
        algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=50)

        self.predictions.clear()

        # Добавление лучших индивидуумов в список предсказаний
        for ind in tools.selBest(population, k=5):
            t = simplify_formula(ind)
            if not self.predictions or self.predictions[-1][1] != t:
                func = toolbox.compile(expr=ind)
                error = evaluate(ind, list(zip(self.x_points, self.y_points)))[0]
                self.predictions.append((func, t, error))

        # Обновление выпадающего списка
        self.prediction_combo.clear()
        for i, _ in enumerate(self.predictions):
            self.prediction_combo.addItem(f"Выражение {i+1}")

        # Отображение первого предсказания
        if self.predictions:
            self.update_prediction(0)
            for i in range(len(self.predictions)):
                print(f'Pred {i+1}: {self.predictions[i][1]}')

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
