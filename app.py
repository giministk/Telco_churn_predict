"""
PyQt5 приложение для прогнозирования оттока клиентов телеком-оператора.

Предоставляет:
- Одиночный прогноз по заполненной форме
- Массовый прогноз из CSV-файла
- Визуализацию важности признаков
- Сохранение отчетов
"""

import sys
import os
import pickle
import csv
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import numpy as np

try:
    import PyQt5

    pyqt_path = os.path.dirname(PyQt5.__file__)
    plugin_path = os.path.join(pyqt_path, 'Qt5', 'plugins')
    if os.path.exists(plugin_path):
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
        print(f"Установлен путь к плагинам: {plugin_path}")
    else:
        alt_path = os.path.join(sys.prefix, 'Library', 'plugins')
        if os.path.exists(alt_path):
            os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = alt_path
            print(f"Установлен путь к плагинам (альтернативный): {alt_path}")
except Exception as e:
    print(f"Ошибка при установке пути к плагинам: {e}")

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QLineEdit, QPushButton, QFrame, QMessageBox,
    QGroupBox, QGridLayout, QTabWidget, QFileDialog, QTableWidget,
    QTableWidgetItem, QHeaderView, QProgressDialog, QDialog, QScrollArea
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# КОНФИГУРАЦИЯ ПРИЛОЖЕНИЯ

@dataclass(frozen=True)
class AppConfig:
    """Конфигурация приложения."""

    # Пути к моделям
    models_dir: str = 'models'

    # Пороги для уровней риска
    high_risk_threshold: float = 0.7
    medium_risk_threshold: float = 0.3

    # Файлы отчетов
    single_report_file: str = 'reports.csv'
    batch_report_default: str = 'batch_predictions.csv'

    # Параметры окна
    window_min_width: int = 1100
    window_min_height: int = 800


# УТИЛИТЫ ДЛЯ РАБОТЫ С МОДЕЛЯМИ

class ModelLoader:
    """Загрузчик моделей и вспомогательных объектов."""

    def __init__(self, config: AppConfig):
        """
        Args:
            config: Конфигурация приложения
        """
        self.config = config
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.feature_importance = None
        self.best_model_name = None
        self.threshold = 0.5
        self.is_loaded = False

    def load(self) -> bool:
        """
        Загружает все необходимые артефакты из директории models.

        Returns:
            True если загрузка успешна, иначе False
        """
        try:
            model_dir = self.config.models_dir

            if not os.path.exists(model_dir):
                print(f"Директория {model_dir} не найдена")
                return False

            # Загрузка модели
            with open(os.path.join(model_dir, 'best_model.pkl'), 'rb') as f:
                self.model = pickle.load(f)

            # Загрузка scaler
            with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
                self.scaler = pickle.load(f)

            # Загрузка имен признаков
            with open(os.path.join(model_dir, 'feature_names.pkl'), 'rb') as f:
                self.feature_names = pickle.load(f)

            # Загрузка важности признаков
            with open(os.path.join(model_dir, 'feature_importance.pkl'), 'rb') as f:
                self.feature_importance = pickle.load(f)

            # Загрузка названия модели (опционально)
            name_path = os.path.join(model_dir, 'best_model_name.pkl')
            if os.path.exists(name_path):
                with open(name_path, 'rb') as f:
                    self.best_model_name = pickle.load(f)
            else:
                self.best_model_name = "Модель"

            # Загрузка порога (опционально)
            threshold_path = os.path.join(model_dir, 'best_threshold.pkl')
            if os.path.exists(threshold_path):
                with open(threshold_path, 'rb') as f:
                    self.threshold = pickle.load(f)

            self.is_loaded = True
            return True

        except Exception as e:
            print(f"Ошибка загрузки моделей: {e}")
            self.is_loaded = False
            return False


# ПРЕДОБРАБОТКА ДАННЫХ ДЛЯ ПРОГНОЗА

class DataPreprocessor:
    """Предобработка данных для прогнозирования."""

    # Отображение бинарных значений с русского на английский
    BINARY_MAP = {'Нет': 0, 'Да': 1}

    # Отображение для SeniorCitizen
    SENIOR_MAP = {'Нет (0)': 0, 'Да (1)': 1}

    @classmethod
    def prepare_features(
            cls,
            data: Dict[str, Any],
            feature_names: List[str],
            scaler: Any
    ) -> np.ndarray:
        """
        Преобразует входные данные в формат, ожидаемый моделью.

        Args:
            data: Словарь с данными клиента
            feature_names: Список всех признаков модели
            scaler: Обученный StandardScaler

        Returns:
            Массив признаков, готовый для predict_proba
        """
        # Создаем DataFrame из одного или нескольких объектов
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()

        # Преобразуем SeniorCitizen
        if 'SeniorCitizen' in df.columns:
            if df['SeniorCitizen'].dtype == 'object':
                df['SeniorCitizen'] = df['SeniorCitizen'].map(cls.SENIOR_MAP)

        # Преобразуем бинарные признаки
        for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
            if col in df.columns:
                df[col] = df[col].map(cls.BINARY_MAP)

        # One-Hot Encoding
        df = pd.get_dummies(df, drop_first=True)

        # Добавляем отсутствующие признаки
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0

        # Приводим к нужному порядку признаков и масштабируем
        X = df[feature_names].values
        return scaler.transform(X)


# МОДЕЛЬ ДАННЫХ ДЛЯ РЕЗУЛЬТАТОВ

@dataclass
class PredictionResult:
    """Результат прогнозирования."""

    probability: float
    prediction_class: int
    risk_level: str
    input_data: Dict[str, Any]

    @classmethod
    def from_prediction(
            cls,
            proba: float,
            threshold: float,
            high_risk: float,
            medium_risk: float,
            input_data: Dict[str, Any]
    ) -> 'PredictionResult':
        """
        Создает объект результата из вероятности.

        Args:
            proba: Вероятность оттока (0-1)
            threshold: Порог классификации
            high_risk: Порог высокого риска
            medium_risk: Порог среднего риска
            input_data: Исходные данные клиента
        """
        pred_class = 1 if proba >= threshold else 0

        # Определяем уровень риска
        if proba >= high_risk:
            risk = "ВЫСОКИЙ"
        elif proba >= medium_risk:
            risk = "СРЕДНИЙ"
        else:
            risk = "НИЗКИЙ"

        return cls(
            probability=proba,
            prediction_class=pred_class,
            risk_level=risk,
            input_data=input_data
        )

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует результат в словарь для сохранения."""
        return {
            **self.input_data,
            'probability': f"{self.probability * 100:.1f}%",
            'prediction': 'Уйдёт' if self.prediction_class == 1 else 'Останется',
            'risk_level': self.risk_level
        }


# ОКНО ВИЗУАЛИЗАЦИИ ВАЖНОСТИ ПРИЗНАКОВ

class FeatureImportanceWindow(QDialog):
    """Окно для отображения важности признаков в виде горизонтальной гистограммы."""

    # Словарь перевода признаков на русский
    RUSSIAN_NAMES = {
        'tenure': 'Стаж (мес.)',
        'Contract_Two year': 'Контракт: 2 года',
        'Contract_One year': 'Контракт: 1 год',
        'MonthlyCharges': 'Ежемесячный платёж',
        'InternetService_Fiber optic': 'Интернет: оптоволокно',
        'OnlineSecurity_Yes': 'Онлайн-безопасность: Да',
        'TechSupport_Yes': 'Техподдержка: Да',
        'OnlineBackup_Yes': 'Онлайн-резерв: Да',
        'DeviceProtection_Yes': 'Защита устройств: Да',
        'StreamingTV_Yes': 'Стриминг TV: Да',
        'StreamingMovies_Yes': 'Стриминг фильмов: Да',
        'PaperlessBilling_Yes': 'Безбумажный счёт: Да',
        'gender_Male': 'Пол: Мужской',
        'SeniorCitizen': 'Пенсионер',
        'Partner_Yes': 'Партнёр: Да',
        'Dependents_Yes': 'Иждивенцы: Да',
        'PhoneService_Yes': 'Телефонная связь: Да',
        'MultipleLines_Yes': 'Несколько линий: Да',
        'MultipleLines_No phone service': 'Несколько линий: нет услуги',
        'InternetService_DSL': 'Интернет: DSL',
        'OnlineSecurity_No internet service': 'Онлайн-безопасность: нет интернета',
        'OnlineBackup_No internet service': 'Онлайн-резерв: нет интернета',
        'DeviceProtection_No internet service': 'Защита устройств: нет интернета',
        'TechSupport_No internet service': 'Техподдержка: нет интернета',
        'StreamingTV_No internet service': 'Стриминг TV: нет интернета',
        'StreamingMovies_No internet service': 'Стриминг фильмов: нет интернета',
        'PaymentMethod_Electronic check': 'Оплата: электронный чек',
        'PaymentMethod_Mailed check': 'Оплата: почтовый чек',
        'PaymentMethod_Bank transfer (automatic)': 'Оплата: банковский перевод',
        'PaymentMethod_Credit card (automatic)': 'Оплата: кредитная карта',
        'TotalCharges': 'Общая сумма платежей',
        'OnlineSecurity_No': 'Онлайн-безопасность: Нет',
        'TechSupport_No': 'Техподдержка: Нет',
        'OnlineBackup_No': 'Онлайн-резерв: Нет',
        'DeviceProtection_No': 'Защита устройств: Нет',
        'StreamingTV_No': 'Стриминг TV: Нет',
        'StreamingMovies_No': 'Стриминг фильмов: Нет',
        'PaperlessBilling_No': 'Безбумажный счёт: Нет',
        'gender_Female': 'Пол: Женский',
        'Partner_No': 'Партнёр: Нет',
        'Dependents_No': 'Иждивенцы: Нет',
        'PhoneService_No': 'Телефонная связь: Нет',
        'MultipleLines_No': 'Несколько линий: Нет',
        'InternetService_No': 'Интернет: Нет',
        'Contract_Month-to-month': 'Контракт: помесячный',
    }

    def __init__(
            self,
            feature_importance: Dict[str, float],
            parent=None
    ):
        """
        Args:
            feature_importance: Словарь {имя_признака: важность}
            parent: Родительское окно
        """
        super().__init__(parent)
        self.feature_importance = feature_importance
        self.setWindowTitle("Важность признаков (Feature Importance)")
        self.setMinimumSize(900, 700)
        self._init_ui()

    def _init_ui(self) -> None:
        """Инициализирует пользовательский интерфейс."""
        layout = QVBoxLayout(self)

        # Заголовок
        title = QLabel("Важность признаков для прогнозирования оттока")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Пояснение
        info = QLabel(
            "Диаграмма показывает, какие характеристики клиента наиболее сильно влияют на прогноз.\n"
            "Чем длиннее столбец, тем больше влияние признака на решение модели."
        )
        info.setAlignment(Qt.AlignCenter)
        info.setStyleSheet("color: #666; font-size: 12px;")
        layout.addWidget(info)

        # Область прокрутки для графика
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.canvas = MplCanvas(self, width=10, height=8, dpi=100)
        scroll_area.setWidget(self.canvas)
        layout.addWidget(scroll_area)

        # Кнопка закрытия
        btn_close = QPushButton("Закрыть")
        btn_close.setStyleSheet("""
            QPushButton {
                background-color: #78909C;
                color: white;
                border-radius: 6px;
                padding: 8px 20px;
            }
            QPushButton:hover {
                background-color: #546E7A;
            }
        """)
        btn_close.clicked.connect(self.close)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(btn_close)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self._draw_plot()

    def _draw_plot(self) -> None:
        """Отрисовывает гистограмму важности признаков."""
        if not self.feature_importance:
            return

        self.canvas.fig.clear()
        ax = self.canvas.fig.add_subplot(111)

        # Сортируем по убыванию важности
        sorted_importance = dict(
            sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        )

        names = list(sorted_importance.keys())
        values = list(sorted_importance.values())

        # Переводим названия на русский
        translated_names = [self.RUSSIAN_NAMES.get(name, name) for name in names]

        # Цвета: выделяем признаки с важностью > 5%
        colors = ['#2196F3' if v > 0.05 else '#78909C' for v in values]

        # Горизонтальная гистограмма
        ax.barh(range(len(values)), values, color=colors)
        ax.set_yticks(range(len(values)))
        ax.set_yticklabels(translated_names, fontsize=9)
        ax.set_xlabel('Важность', fontsize=12)
        ax.set_title('Важность признаков (Random Forest)', fontsize=14, fontweight='bold')
        ax.invert_yaxis()

        # Подписи значений
        for i, v in enumerate(values):
            ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=8)

        ax.set_xlim(0, max(values) * 1.15)
        self.canvas.fig.tight_layout()
        self.canvas.draw()


# ОСНОВНОЕ ОКНО ПРИЛОЖЕНИЯ

class MainWindow(QMainWindow):
    """Главное окно приложения для прогнозирования оттока."""

    def __init__(self):
        super().__init__()

        # Состояние приложения
        self.config = AppConfig()
        self.model_loader = ModelLoader(self.config)
        self.preprocessor = DataPreprocessor()

        # Данные
        self.fields_data: Dict[str, Any] = {}
        self.last_result: Optional[PredictionResult] = None
        self.batch_data: Optional[pd.DataFrame] = None
        self.batch_results: Optional[List[Dict[str, Any]]] = None

        # Настройка окна
        self.setWindowTitle("Приложение для прогнозирования оттока клиентов")
        self.setMinimumSize(
            self.config.window_min_width,
            self.config.window_min_height
        )

        # Загрузка моделей
        self._load_models()

        # Инициализация UI
        self._init_ui()

        # Если модели не загружены, показываем ошибку
        if not self.model_loader.is_loaded:
            QTimer.singleShot(500, self._show_load_error)

    # ЗАГРУЗКА МОДЕЛЕЙ

    def _load_models(self) -> None:
        """Загружает модели и обновляет состояние интерфейса."""
        self.model_loader.load()
        if not self.model_loader.is_loaded:
            print("Модели не загружены. Запустите train_models_ibm.py сначала.")

    def _show_load_error(self) -> None:
        """Показывает диалог ошибки загрузки моделей."""
        if not self.model_loader.is_loaded:
            QMessageBox.warning(
                self,
                "Ошибка загрузки моделей",
                "Не удалось загрузить модели.\n\n"
                "Пожалуйста, сначала выполните скрипт train_models_ibm.py\n"
                "для обучения и сохранения моделей в папку 'models'."
            )

    # ИНИЦИАЛИЗАЦИЯ UI

    def _init_ui(self) -> None:
        """Создает пользовательский интерфейс."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Заголовок
        title_label = QLabel("Прогнозирование оттока клиентов")
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        # Информация о модели
        self.model_info_label = self._create_model_info_label()
        main_layout.addWidget(self.model_info_label)

        # Вкладки
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Создаем вкладки
        self._create_single_prediction_tab()
        self._create_batch_prediction_tab()

        # Обновляем информацию
        self._update_model_info()

    def _create_model_info_label(self) -> QLabel:
        """Создает виджет с информацией о модели."""
        label = QLabel("Модель: Загрузка...")
        label.setAlignment(Qt.AlignCenter)
        label.setFont(QFont("Arial", 13, QFont.Bold))
        label.setStyleSheet(
            "color: #1976D2; background-color: #E3F2FD; padding: 8px; border-radius: 6px;"
        )
        label.setToolTip(
            "Краткое описание модели:\n\n"
            "• Логистическая регрессия — линейная модель,\n"
            "  оценивающая вероятность с помощью сигмоидной функции.\n\n"
            "• Случайный лес — ансамбль решающих деревьев,\n"
            "  построенных на случайных подвыборках.\n\n"
            "• Градиентный бустинг — последовательное построение\n"
            "  деревьев, каждое из которых исправляет ошибки предыдущего.\n\n"
            " Выбрана модель с наилучшим F1-score на тестовых данных."
        )
        return label

    def _update_model_info(self) -> None:
        """Обновляет информацию о модели в заголовке."""
        if self.model_loader.is_loaded:
            threshold = self.model_loader.threshold
            name = self.model_loader.best_model_name
            self.model_info_label.setText(f"Модель: {name} | Порог: {threshold:.3f}")

    # ВКЛАДКА: ОДИНОЧНЫЙ ПРОГНОЗ

    def _create_single_prediction_tab(self) -> None:
        """Создает вкладку для одиночного прогноза."""
        tab = QWidget()
        layout = QHBoxLayout(tab)

        # Левая панель - форма ввода
        left_panel = self._create_input_panel()
        layout.addWidget(left_panel, 1)

        # Правая панель - результаты
        right_panel = self._create_results_panel()
        layout.addWidget(right_panel, 1)

        self.tab_widget.addTab(tab, "Одиночный прогноз")

    def _create_input_panel(self) -> QFrame:
        """Создает панель с формой ввода данных клиента."""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        panel.setMinimumWidth(450)
        panel_layout = QVBoxLayout(panel)

        # Группа полей ввода
        input_group = QGroupBox("Данные клиента")
        input_group.setFont(QFont("Arial", 11, QFont.Bold))

        grid = self._create_input_grid()
        input_group.setLayout(grid)
        panel_layout.addWidget(input_group)

        # Кнопки управления
        button_layout = self._create_control_buttons()
        panel_layout.addLayout(button_layout)

        # Статус
        self.status_label = QLabel(
            "Готов к работе" if self.model_loader.is_loaded else "Загрузка моделей..."
        )
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 12px; color: #666;")
        panel_layout.addWidget(self.status_label)

        return panel

    def _create_input_grid(self) -> QGridLayout:
        """Создает сетку полей ввода."""
        grid = QGridLayout()
        grid.setVerticalSpacing(8)
        grid.setHorizontalSpacing(15)

        self.fields = {}

        # Определяем поля: (имя, тип, значения, индекс по умолчанию)
        field_defs = [
            ('gender', 'combo', ['Female', 'Male'], 0),
            ('SeniorCitizen', 'combo', ['Нет (0)', 'Да (1)'], 0),
            ('Partner', 'combo', ['Нет', 'Да'], 0),
            ('Dependents', 'combo', ['Нет', 'Да'], 0),
            ('tenure', 'input', 'Например: 12', '12'),
            ('PhoneService', 'combo', ['Нет', 'Да'], 1),
            ('MultipleLines', 'combo', ['No phone service', 'No', 'Yes'], 1),
            ('InternetService', 'combo', ['No', 'DSL', 'Fiber optic'], 1),
            ('OnlineSecurity', 'combo', ['No internet service', 'No', 'Yes'], 1),
            ('OnlineBackup', 'combo', ['No internet service', 'No', 'Yes'], 1),
            ('DeviceProtection', 'combo', ['No internet service', 'No', 'Yes'], 1),
            ('TechSupport', 'combo', ['No internet service', 'No', 'Yes'], 1),
            ('StreamingTV', 'combo', ['No internet service', 'No', 'Yes'], 1),
            ('StreamingMovies', 'combo', ['No internet service', 'No', 'Yes'], 1),
            ('Contract', 'combo', ['Month-to-month', 'One year', 'Two year'], 0),
            ('PaperlessBilling', 'combo', ['Нет', 'Да'], 0),
            ('PaymentMethod', 'combo', [
                'Electronic check', 'Mailed check',
                'Bank transfer (automatic)', 'Credit card (automatic)'
            ], 0),
            ('MonthlyCharges', 'input', 'Например: 70.50', '70.50'),
            ('TotalCharges', 'input', 'Например: 1500.00', '1500.00'),
        ]

        labels = {
            'gender': 'Пол:',
            'SeniorCitizen': 'Пенсионер:',
            'Partner': 'Партнёр:',
            'Dependents': 'Иждивенцы:',
            'tenure': 'Стаж (месяцев):',
            'PhoneService': 'Телефонная связь:',
            'MultipleLines': 'Несколько линий:',
            'InternetService': 'Интернет-услуга:',
            'OnlineSecurity': 'Онлайн-безопасность:',
            'OnlineBackup': 'Онлайн-резерв:',
            'DeviceProtection': 'Защита устройств:',
            'TechSupport': 'Техподдержка:',
            'StreamingTV': 'Стриминг TV:',
            'StreamingMovies': 'Стриминг фильмов:',
            'Contract': 'Тип контракта:',
            'PaperlessBilling': 'Безбумажный счёт:',
            'PaymentMethod': 'Способ оплаты:',
            'MonthlyCharges': 'Ежемесячный платёж:',
            'TotalCharges': 'Общая сумма:',
        }

        for row, (name, field_type, *args) in enumerate(field_defs):
            grid.addWidget(QLabel(labels[name]), row, 0)

            if field_type == 'combo':
                values, default_idx = args
                combo = QComboBox()
                combo.addItems(values)
                combo.setCurrentIndex(default_idx)
                self.fields[name] = combo
                grid.addWidget(combo, row, 1)
            else:  # 'input'
                placeholder, default_value = args
                line_edit = QLineEdit()
                line_edit.setPlaceholderText(placeholder)
                line_edit.setText(default_value)
                self.fields[name] = line_edit
                grid.addWidget(line_edit, row, 1)

        return grid

    def _create_control_buttons(self) -> QHBoxLayout:
        """Создает кнопки управления прогнозом."""
        layout = QHBoxLayout()

        # Кнопка прогноза
        self.predict_btn = QPushButton("Получить прогноз")
        self.predict_btn.setFont(QFont("Arial", 12, QFont.Bold))
        self.predict_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border-radius: 8px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #B0BEC5;
            }
        """)
        self.predict_btn.clicked.connect(self._on_predict_clicked)
        self.predict_btn.setEnabled(self.model_loader.is_loaded)
        layout.addWidget(self.predict_btn)

        # Кнопка очистки
        self.clear_btn = QPushButton("Очистить")
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #78909C;
                color: white;
                border-radius: 8px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #546E7A;
            }
        """)
        self.clear_btn.clicked.connect(self._clear_fields)
        layout.addWidget(self.clear_btn)

        return layout

    def _create_results_panel(self) -> QFrame:
        """Создает панель с результатами прогноза."""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        panel.setMinimumWidth(400)
        panel_layout = QVBoxLayout(panel)

        # Группа результата
        result_group = QGroupBox("Результат прогноза")
        result_group.setFont(QFont("Arial", 11, QFont.Bold))
        result_layout = QVBoxLayout()

        # Вероятность
        self.prob_label = QLabel("Вероятность оттока:")
        self.prob_label.setFont(QFont("Arial", 12))
        self.prob_value = QLabel("—")
        self.prob_value.setFont(QFont("Arial", 28, QFont.Bold))
        self.prob_value.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(self.prob_label)
        result_layout.addWidget(self.prob_value)

        # Прогноз
        self.pred_label = QLabel("Прогноз:")
        self.pred_label.setFont(QFont("Arial", 12))
        self.pred_value = QLabel("—")
        self.pred_value.setFont(QFont("Arial", 18, QFont.Bold))
        self.pred_value.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(self.pred_label)
        result_layout.addWidget(self.pred_value)

        # Уровень риска
        self.risk_label = QLabel("Уровень риска:")
        self.risk_label.setFont(QFont("Arial", 12))
        self.risk_value = QLabel("—")
        self.risk_value.setFont(QFont("Arial", 16, QFont.Bold))
        self.risk_value.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(self.risk_label)
        result_layout.addWidget(self.risk_value)

        result_group.setLayout(result_layout)
        panel_layout.addWidget(result_group)

        # Кнопка важности признаков
        btn_importance = QPushButton("Показать важность признаков")
        btn_importance.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border-radius: 6px;
                padding: 8px 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #E68900;
            }
            QPushButton:disabled {
                background-color: #B0BEC5;
            }
        """)
        btn_importance.clicked.connect(self._show_feature_importance)
        btn_importance.setEnabled(self.model_loader.is_loaded)
        panel_layout.addWidget(btn_importance)

        # Кнопка сохранения отчета
        self.save_btn = QPushButton("Сохранить отчёт")
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 8px;
                padding: 10px 15px;
            }
            QPushButton:hover {
                background-color: #388E3C;
            }
            QPushButton:disabled {
                background-color: #B0BEC5;
            }
        """)
        self.save_btn.clicked.connect(self._on_save_clicked)
        self.save_btn.setEnabled(False)
        panel_layout.addWidget(self.save_btn)

        return panel

    # ВКЛАДКА: МАССОВОЕ ПРОГНОЗИРОВАНИЕ

    def _create_batch_prediction_tab(self) -> None:
        """Создает вкладку для массового прогнозирования из CSV."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Панель управления
        control_group = QGroupBox("Загрузка CSV-файла")
        control_group.setFont(QFont("Arial", 11, QFont.Bold))
        control_layout = QHBoxLayout()

        # Кнопки
        self.load_csv_btn = QPushButton("Загрузить CSV-файл")
        self.load_csv_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border-radius: 8px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #E68900;
            }
            QPushButton:disabled {
                background-color: #B0BEC5;
            }
        """)
        self.load_csv_btn.clicked.connect(self._load_csv_file)
        self.load_csv_btn.setEnabled(self.model_loader.is_loaded)
        control_layout.addWidget(self.load_csv_btn)

        self.batch_predict_btn = QPushButton("Выполнить прогноз для всех")
        self.batch_predict_btn.setStyleSheet("""
            QPushButton {
                background-color: #4169E1;
                color: white;
                border-radius: 8px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #0000CD;
            }
            QPushButton:disabled {
                background-color: #B0BEC5;
            }
        """)
        self.batch_predict_btn.clicked.connect(self._on_batch_predict)
        self.batch_predict_btn.setEnabled(False)
        control_layout.addWidget(self.batch_predict_btn)

        self.batch_save_btn = QPushButton("Сохранить результаты")
        self.batch_save_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 8px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #388E3C;
            }
            QPushButton:disabled {
                background-color: #B0BEC5;
            }
        """)
        self.batch_save_btn.clicked.connect(self._save_batch_results)
        self.batch_save_btn.setEnabled(False)
        control_layout.addWidget(self.batch_save_btn)

        # Статус
        self.batch_status_label = QLabel("Загрузите CSV-файл с данными клиентов")
        self.batch_status_label.setAlignment(Qt.AlignCenter)
        self.batch_status_label.setStyleSheet("color: #666; font-size: 12px;")
        control_layout.addWidget(self.batch_status_label)

        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        # Таблица для отображения данных
        self.batch_table = QTableWidget()
        self.batch_table.setAlternatingRowColors(False)
        self.batch_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        layout.addWidget(self.batch_table)

        self.tab_widget.addTab(tab, "Массовый прогноз (CSV)")

    # ОБРАБОТЧИКИ СОБЫТИЙ

    def _clear_fields(self) -> None:
        """Очищает все поля ввода и результаты."""
        for widget in self.fields.values():
            if isinstance(widget, QLineEdit):
                widget.clear()

        # Сбрасываем результаты
        self.prob_value.setText("—")
        self.pred_value.setText("—")
        self.risk_value.setText("—")
        self.prob_value.setStyleSheet("color: #333;")
        self.pred_value.setStyleSheet("color: #333;")
        self.risk_value.setStyleSheet("color: #333;")
        self.save_btn.setEnabled(False)
        self.last_result = None

    def _validate_input(self) -> List[str]:
        """
        Проверяет корректность введенных данных.

        Returns:
            Список ошибок (пустой если все корректно)
        """
        errors = []

        for name, widget in self.fields.items():
            if isinstance(widget, QLineEdit):
                text = widget.text().strip()
                if not text:
                    errors.append(f"Поле '{name}' не заполнено")
                    continue

                try:
                    value = float(text)
                    if value < 0:
                        errors.append(f"Значение поля '{name}' не может быть отрицательным")
                except ValueError:
                    errors.append(f"В поле '{name}' необходимо ввести число")

        return errors

    def _get_input_data(self) -> Dict[str, Any]:
        """Собирает данные из полей ввода."""
        data = {}
        for name, widget in self.fields.items():
            if isinstance(widget, QComboBox):
                data[name] = widget.currentText()
            elif isinstance(widget, QLineEdit):
                data[name] = float(widget.text().strip())
        return data

    def _display_result(self, result: PredictionResult) -> None:
        """
        Отображает результат прогноза в интерфейсе.

        Args:
            result: Объект с результатом прогноза
        """
        proba = result.probability
        proba_percent = proba * 100

        # Вероятность
        if proba >= self.config.high_risk_threshold:
            color = "#D32F2F"
        elif proba >= self.config.medium_risk_threshold:
            color = "#F9A825"
        else:
            color = "#388E3C"

        self.prob_value.setText(f"{proba_percent:.1f}%")
        self.prob_value.setStyleSheet(f"color: {color}; font-size: 28px; font-weight: bold;")

        # Прогноз
        if result.prediction_class == 1:
            pred_text = "Клиент уйдёт"
            pred_color = "#D32F2F"
        else:
            pred_text = "Клиент останется"
            pred_color = "#388E3C"

        self.pred_value.setText(pred_text)
        self.pred_value.setStyleSheet(f"color: {pred_color}; font-size: 18px; font-weight: bold;")

        # Уровень риска
        risk_colors = {
            "ВЫСОКИЙ": "#D32F2F",
            "СРЕДНИЙ": "#F9A825",
            "НИЗКИЙ": "#388E3C"
        }
        self.risk_value.setText(result.risk_level)
        self.risk_value.setStyleSheet(
            f"color: {risk_colors[result.risk_level]}; font-size: 16px; font-weight: bold;"
        )

    def _on_predict_clicked(self) -> None:
        """Обработчик нажатия кнопки 'Получить прогноз'."""
        if not self.model_loader.is_loaded:
            QMessageBox.warning(self, "Ошибка", "Модели не загружены")
            return

        # Валидация
        errors = self._validate_input()
        if errors:
            QMessageBox.warning(self, "Ошибка ввода", "\n".join(errors))
            return

        try:
            data = self._get_input_data()
            self.fields_data = data

            # Подготовка признаков
            X_scaled = DataPreprocessor.prepare_features(
                data,
                self.model_loader.feature_names,
                self.model_loader.scaler
            )

            # Прогноз
            proba = self.model_loader.model.predict_proba(X_scaled)[0][1]

            # Создание результата
            result = PredictionResult.from_prediction(
                proba,
                self.model_loader.threshold,
                self.config.high_risk_threshold,
                self.config.medium_risk_threshold,
                data
            )

            self.last_result = result
            self._display_result(result)
            self.save_btn.setEnabled(True)

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Ошибка", f"Произошла ошибка при прогнозировании:\n{str(e)}")

    def _on_save_clicked(self) -> None:
        """Обработчик нажатия кнопки 'Сохранить отчёт'."""
        if self.last_result is None:
            QMessageBox.warning(self, "Ошибка", "Сначала выполните прогноз")
            return

        report_data = self._create_report_data(self.last_result)
        success, msg = self._save_report_to_csv(report_data)

        if success:
            QMessageBox.information(self, "Успех", msg)
        else:
            QMessageBox.warning(self, "Ошибка", f"Не удалось сохранить:\n{msg}")

    def _create_report_data(self, result: PredictionResult) -> Dict[str, Any]:
        """
        Создает словарь с данными для отчета.

        Args:
            result: Результат прогноза

        Returns:
            Словарь с данными для CSV
        """
        data = result.input_data
        return {
            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'gender': data.get('gender', ''),
            'SeniorCitizen': data.get('SeniorCitizen', ''),
            'Partner': data.get('Partner', ''),
            'Dependents': data.get('Dependents', ''),
            'tenure': data.get('tenure', ''),
            'PhoneService': data.get('PhoneService', ''),
            'MultipleLines': data.get('MultipleLines', ''),
            'InternetService': data.get('InternetService', ''),
            'OnlineSecurity': data.get('OnlineSecurity', ''),
            'OnlineBackup': data.get('OnlineBackup', ''),
            'DeviceProtection': data.get('DeviceProtection', ''),
            'TechSupport': data.get('TechSupport', ''),
            'StreamingTV': data.get('StreamingTV', ''),
            'StreamingMovies': data.get('StreamingMovies', ''),
            'Contract': data.get('Contract', ''),
            'PaperlessBilling': data.get('PaperlessBilling', ''),
            'PaymentMethod': data.get('PaymentMethod', ''),
            'MonthlyCharges': data.get('MonthlyCharges', ''),
            'TotalCharges': data.get('TotalCharges', ''),
            'probability': f"{result.probability * 100:.1f}%",
            'prediction': 'Уйдёт' if result.prediction_class == 1 else 'Останется',
            'risk_level': result.risk_level
        }

    def _save_report_to_csv(self, report_data: Dict[str, Any], filename: str = 'reports.csv') -> Tuple[bool, str]:
        """
        Сохраняет отчет в CSV-файл.

        Args:
            report_data: Данные для сохранения
            filename: Имя файла

        Returns:
            Кортеж (успех, сообщение)
        """
        try:
            file_exists = os.path.isfile(filename)

            fieldnames = [
                'date', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
                'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                'PaymentMethod', 'MonthlyCharges', 'TotalCharges',
                'probability', 'prediction', 'risk_level'
            ]

            with open(filename, mode='a', newline='', encoding='utf-8-sig') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=';')
                if not file_exists:
                    writer.writeheader()
                writer.writerow(report_data)

            return True, f"Сохранено в {filename}"
        except Exception as e:
            return False, str(e)

    def _show_feature_importance(self) -> None:
        """Показывает окно с важностью признаков."""
        if not self.model_loader.feature_importance:
            QMessageBox.warning(self, "Ошибка", "Данные о важности признаков не загружены")
            return

        window = FeatureImportanceWindow(self.model_loader.feature_importance, self)
        window.exec_()

    # МАССОВОЕ ПРОГНОЗИРОВАНИЕ (ОБРАБОТЧИКИ)

    def _load_csv_file(self) -> None:
        """Загружает CSV-файл с данными для массового прогнозирования."""
        if not self.model_loader.is_loaded:
            QMessageBox.warning(self, "Ошибка", "Модели не загружены")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите CSV-файл", "", "CSV файлы (*.csv)"
        )

        if not file_path:
            return

        try:
            df = pd.read_csv(file_path)

            # Проверяем наличие обязательных столбцов
            required_cols = [
                'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                'MonthlyCharges', 'TotalCharges'
            ]

            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                QMessageBox.warning(
                    self,
                    "Ошибка",
                    f"В файле отсутствуют необходимые столбцы:\n{', '.join(missing_cols)}\n\n"
                    "Убедитесь, что файл содержит все признаки датасета IBM Telco."
                )
                return

            self.batch_data = df
            self.batch_results = None

            self._display_batch_data(df)

            self.batch_status_label.setText(f"Загружено {len(df)} записей. Нажмите 'Выполнить прогноз для всех'.")
            self.batch_predict_btn.setEnabled(True)
            self.batch_save_btn.setEnabled(False)

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить файл:\n{str(e)}")

    def _display_batch_data(self, df: pd.DataFrame) -> None:
        """Отображает загруженные данные в таблице."""
        self.batch_table.setRowCount(len(df))
        self.batch_table.setColumnCount(len(df.columns))
        self.batch_table.setHorizontalHeaderLabels(df.columns)

        for i, row in df.iterrows():
            for j, value in enumerate(row):
                self.batch_table.setItem(i, j, QTableWidgetItem(str(value)))

        self.batch_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

    def _on_batch_predict(self) -> None:
        """Выполняет прогнозирование для всех записей из CSV."""
        if self.batch_data is None:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите CSV-файл")
            return

        try:
            progress = QProgressDialog(
                "Выполнение прогнозов...", "Отмена",
                0, len(self.batch_data), self
            )
            progress.setWindowModality(Qt.WindowModal)
            progress.show()

            results = []

            for idx, row in self.batch_data.iterrows():
                if progress.wasCanceled():
                    break

                progress.setValue(idx)

                data = row.to_dict()

                # Подготовка признаков
                X_scaled = DataPreprocessor.prepare_features(
                    data,
                    self.model_loader.feature_names,
                    self.model_loader.scaler
                )

                # Прогноз
                proba = self.model_loader.model.predict_proba(X_scaled)[0][1]

                result = PredictionResult.from_prediction(
                    proba,
                    self.model_loader.threshold,
                    self.config.high_risk_threshold,
                    self.config.medium_risk_threshold,
                    data
                )

                results.append({
                    'index': idx,
                    **result.to_dict()
                })

            progress.setValue(len(self.batch_data))

            self.batch_results = results
            self._display_batch_results(results)

            self.batch_status_label.setText(f"Прогноз выполнен для {len(results)} записей")
            self.batch_save_btn.setEnabled(True)

            QMessageBox.information(
                self,
                "Прогноз завершён",
                "Результаты прогноза отображаются в правой части таблицы."
            )

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при выполнении прогнозов:\n{str(e)}")

    def _display_batch_results(self, results: List[Dict[str, Any]]) -> None:
        """Отображает результаты массового прогноза в таблице."""
        columns = list(self.batch_data.columns) + ['Вероятность_оттока', 'Прогноз', 'Уровень_риска']

        self.batch_table.setRowCount(len(results))
        self.batch_table.setColumnCount(len(columns))
        self.batch_table.setHorizontalHeaderLabels(columns)

        for i, result in enumerate(results):
            # Цвет строки в зависимости от прогноза
            if result['prediction'] == 'Уйдёт':
                row_color = QColor(255, 200, 200)
            else:
                row_color = QColor(200, 255, 200)

            # Заполняем исходные данные
            col_idx = 0
            for key in self.batch_data.columns:
                item = QTableWidgetItem(str(result.get(key, '')))
                item.setBackground(row_color)
                self.batch_table.setItem(i, col_idx, item)
                col_idx += 1

            # Вероятность
            item = QTableWidgetItem(result['probability'])
            item.setBackground(row_color)
            self.batch_table.setItem(i, col_idx, item)
            col_idx += 1

            # Прогноз
            item = QTableWidgetItem(result['prediction'])
            item.setBackground(row_color)
            self.batch_table.setItem(i, col_idx, item)
            col_idx += 1

            # Уровень риска
            item = QTableWidgetItem(result['risk_level'])
            item.setBackground(row_color)
            self.batch_table.setItem(i, col_idx, item)

        self.batch_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

    def _save_batch_results(self) -> None:
        """Сохраняет результаты массового прогноза в CSV."""
        if self.batch_results is None:
            QMessageBox.warning(self, "Ошибка", "Сначала выполните прогноз")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить результаты", self.config.batch_report_default, "CSV файлы (*.csv)"
        )

        if not file_path:
            return

        try:
            df_results = pd.DataFrame(self.batch_results)
            df_results.to_csv(file_path, index=False, encoding='utf-8-sig')
            QMessageBox.information(self, "Успех", f"Результаты сохранены в:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить файл:\n{str(e)}")


# ВСПОМОГАТЕЛЬНЫЙ КЛАСС ДЛЯ МАТПЛОТЛИБ

class MplCanvas(FigureCanvas):
    """Canvas для отображения графиков Matplotlib в Qt."""

    def __init__(self, parent=None, width=6, height=5, dpi=100):
        """
        Args:
            parent: Родительский виджет
            width: Ширина фигуры в дюймах
            height: Высота фигуры в дюймах
            dpi: Разрешение
        """
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)


# ТОЧКА ВХОДА

def configure_qt_plugins():
    """
    Настраивает путь к плагинам Qt для корректной работы на некоторых системах.
    """
    try:
        import PyQt5
        pyqt_path = os.path.dirname(PyQt5.__file__)
        plugin_path = os.path.join(pyqt_path, 'Qt5', 'plugins')

        if os.path.exists(plugin_path):
            os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
            print(f"Установлен путь к плагинам: {plugin_path}")
        else:
            alt_path = os.path.join(sys.prefix, 'Library', 'plugins')
            if os.path.exists(alt_path):
                os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = alt_path
                print(f"Установлен путь к плагинам (альтернативный): {alt_path}")
    except Exception as e:
        print(f"Ошибка при установке пути к плагинам: {e}")


def setup_matplotlib():
    """Настраивает Matplotlib для использования Qt5 бэкенда."""
    import matplotlib
    matplotlib.use('Qt5Agg')


def main():
    """Основная функция запуска приложения."""
    # Настройка окружения
    configure_qt_plugins()
    setup_matplotlib()

    # Создание приложения
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Настройка палитры
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(248, 249, 250))
    palette.setColor(QPalette.WindowText, QColor(33, 33, 33))
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ToolTipText, QColor(33, 33, 33))
    palette.setColor(QPalette.Text, QColor(33, 33, 33))
    palette.setColor(QPalette.Button, QColor(241, 243, 244))
    palette.setColor(QPalette.ButtonText, QColor(33, 33, 33))
    palette.setColor(QPalette.BrightText, QColor(255, 255, 255))
    palette.setColor(QPalette.Highlight, QColor(33, 150, 243))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    # Запуск главного окна
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()