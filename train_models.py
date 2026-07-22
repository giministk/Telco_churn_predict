"""
Модуль для обучения моделей прогнозирования оттока клиентов телеком-оператора.

Содержит функции для загрузки данных, предобработки, обучения нескольких моделей
и сохранения лучшей модели для использования в PyQt5 приложении.
"""

import pandas as pd
import numpy as np
import pickle
import os
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)


# КОНФИГУРАЦИЯ (вынесена для удобства изменения)

@dataclass(frozen=True)
class ModelConfig:
    """Конфигурация параметров обучения моделей."""

    # Параметры разделения данных
    test_size: float = 0.2
    random_state: int = 42

    # Параметры Random Forest
    rf_n_estimators: int = 150
    rf_max_depth: int = 15
    rf_min_samples_split: int = 5
    rf_min_samples_leaf: int = 2

    # Параметры Gradient Boosting
    gb_n_estimators: int = 150
    gb_learning_rate: float = 0.1
    gb_max_depth: int = 5
    gb_min_samples_split: int = 5
    gb_min_samples_leaf: int = 3
    gb_subsample: float = 0.8

    # Параметры логистической регрессии
    lr_max_iter: int = 1000

    # Параметры сохранения
    output_dir: str = 'models'
    data_file: str = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'


# ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ

def load_data(file_path: str = 'WA_Fn-UseC_-Telco-Customer-Churn.csv') -> pd.DataFrame:
    """
    Загружает данные из CSV файла.

    Args:
        file_path: Путь к файлу с данными

    Returns:
        DataFrame с загруженными данными

    Raises:
        FileNotFoundError: Если файл не найден
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл данных не найден: {file_path}")

    df = pd.read_csv(file_path)
    print(f"Загружено {len(df)} записей, {len(df.columns)} столбцов")
    return df


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Выполняет полную предобработку данных для моделирования.

    Этапы:
    1. Удаление идентификатора клиента
    2. Обработка TotalCharges (пропуски → медиана)
    3. Кодирование целевой переменной Churn
    4. Кодирование бинарных признаков
    5. One-Hot Encoding для категориальных признаков

    Args:
        df: Исходный DataFrame

    Returns:
        Кортеж (X - признаки, y - целевая переменная, feature_names - список признаков)
    """
    data = df.copy()

    # Удаляем идентификатор (не несет предсказательной силы)
    data = data.drop(columns=['customerID'])

    # Преобразуем TotalCharges в число, пропуски заполняем медианой
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median())

    # Кодируем целевую переменную
    data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})

    # Кодируем бинарные признаки (Yes=1, No=0)
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        if col in data.columns:
            data[col] = data[col].map({'Yes': 1, 'No': 0})

    # One-Hot Encoding для категориальных признаков (drop_first для избежания мультиколлинеарности)
    categorical_cols = [
        'gender', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaymentMethod'
    ]
    categorical_cols = [col for col in categorical_cols if col in data.columns]
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    # Разделяем признаки и целевую переменную
    y = data['Churn'].values
    X = data.drop(columns=['Churn'])
    feature_names = X.columns.tolist()

    print(f"Предобработка завершена: {len(feature_names)} признаков")
    print(f"Баланс классов: 0={np.sum(y == 0)}, 1={np.sum(y == 1)}")

    return X, y, feature_names


# ОБУЧЕНИЕ И ОЦЕНКА МОДЕЛЕЙ

def split_and_scale_data(
        X: pd.DataFrame,
        y: np.ndarray,
        config: ModelConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Разделяет данные на train/test и применяет стандартизацию.

    Args:
        X: Признаки
        y: Целевая переменная
        config: Конфигурация модели

    Returns:
        Кортеж (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
    """
    # Стратифицированное разделение для сохранения пропорции классов
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y
    )

    # Стандартизация (fit только на train, transform на test)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Данные разделены: train={len(X_train)}, test={len(X_test)}")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def find_optimal_threshold(
        model: Any,
        X_val: np.ndarray,
        y_val: np.ndarray
) -> float:
    """
    Находит оптимальный порог классификации по максимуму F1-меры.

    Args:
        model: Обученная модель с методом predict_proba
        X_val: Валидационные признаки
        y_val: Целевая переменная

    Returns:
        Оптимальный порог (0.0-1.0)
    """
    from sklearn.metrics import precision_recall_curve

    y_proba = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)

    # Вычисляем F1 для всех порогов (добавляем epsilon для избежания деления на 0)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    return best_threshold


def evaluate_model(
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray
) -> Dict[str, float]:
    """
    Вычисляет все метрики качества для модели.

    Args:
        model: Обученная модель
        X_test: Тестовые признаки
        y_test: Тестовые ответы

    Returns:
        Словарь с метриками: accuracy, precision, recall, f1, roc_auc
    """
    y_proba = model.predict_proba(X_test)[:, 1]

    # Находим оптимальный порог
    threshold = find_optimal_threshold(model, X_test, y_test)
    y_pred = (y_proba >= threshold).astype(int)

    return {
        'threshold': threshold,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }


def train_single_model(
        model: Any,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
) -> Dict[str, Any]:
    """
    Обучает одну модель и возвращает результаты.

    Args:
        model: Экземпляр модели sklearn
        model_name: Название модели для логирования
        X_train, y_train: Обучающие данные
        X_test, y_test: Тестовые данные

    Returns:
        Словарь с моделью и метриками
    """
    print(f"\nОбучение {model_name}...")
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)

    # Логируем результаты
    print(f"F1-score: {metrics['f1']:.4f}")
    print(f"ROC-AUC:  {metrics['roc_auc']:.4f}")

    result = {
        'model': model,
        **metrics  # Распаковываем все метрики
    }

    # Добавляем важность признаков для некоторых моделей
    if hasattr(model, 'feature_importances_'):
        result['feature_importance'] = model.feature_importances_

    return result


def train_models(
        X: pd.DataFrame,
        y: np.ndarray,
        config: ModelConfig
) -> Tuple[Dict[str, Any], StandardScaler]:
    """
    Обучает все модели (Логистическая регрессия, Random Forest, Gradient Boosting).

    Args:
        X: Признаки
        y: Целевая переменная
        config: Конфигурация

    Returns:
        Кортеж (results - словарь с результатами всех моделей, scaler)
    """
    # Подготовка данных
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y, config)

    results = {}

    # 1. Логистическая регрессия (базовая модель)
    lr = LogisticRegression(
        max_iter=config.lr_max_iter,
        random_state=config.random_state,
        class_weight='balanced'
    )
    results['LogisticRegression'] = train_single_model(
        lr, 'LogisticRegression', X_train, y_train, X_test, y_test
    )

    # 2. Random Forest (ансамбль деревьев)
    rf = RandomForestClassifier(
        n_estimators=config.rf_n_estimators,
        max_depth=config.rf_max_depth,
        min_samples_split=config.rf_min_samples_split,
        min_samples_leaf=config.rf_min_samples_leaf,
        class_weight='balanced',
        random_state=config.random_state,
        n_jobs=-1
    )
    results['RandomForest'] = train_single_model(
        rf, 'RandomForest', X_train, y_train, X_test, y_test
    )

    # 3. Gradient Boosting (бустинг)
    gb = GradientBoostingClassifier(
        n_estimators=config.gb_n_estimators,
        learning_rate=config.gb_learning_rate,
        max_depth=config.gb_max_depth,
        min_samples_split=config.gb_min_samples_split,
        min_samples_leaf=config.gb_min_samples_leaf,
        subsample=config.gb_subsample,
        random_state=config.random_state
    )
    results['GradientBoosting'] = train_single_model(
        gb, 'GradientBoosting', X_train, y_train, X_test, y_test
    )

    return results, scaler


# АНАЛИЗ И СОХРАНЕНИЕ

def get_best_model(results: Dict[str, Any], metric: str = 'f1') -> str:
    """
    Выбирает лучшую модель по указанной метрике.

    Args:
        results: Словарь с результатами моделей
        metric: Метрика для сравнения (по умолчанию 'f1')

    Returns:
        Название лучшей модели
    """
    best_model = max(results, key=lambda x: results[x][metric])
    print(f"\nЛучшая модель: {best_model} ({metric} = {results[best_model][metric]:.4f})")
    return best_model


def save_model_artifacts(
        results: Dict[str, Any],
        scaler: StandardScaler,
        feature_names: List[str],
        best_model_name: str,
        output_dir: str = 'models'
) -> None:
    """
    Сохраняет все обученные модели и вспомогательные объекты.

    Сохраняет:
    - best_model.pkl: Лучшая модель
    - best_threshold.pkl: Оптимальный порог для лучшей модели
    - scaler.pkl: Обученный StandardScaler
    - feature_names.pkl: Имена признаков
    - {model_name}_model.pkl: Все модели
    - feature_importance.pkl: Важность признаков (если доступна)

    Args:
        results: Словарь с результатами моделей
        scaler: Обученный StandardScaler
        feature_names: Список имен признаков
        best_model_name: Название лучшей модели
        output_dir: Директория для сохранения
    """
    os.makedirs(output_dir, exist_ok=True)

    # Сохраняем лучшую модель
    best_model = results[best_model_name]
    with open(os.path.join(output_dir, 'best_model.pkl'), 'wb') as f:
        pickle.dump(best_model['model'], f)

    # Сохраняем оптимальный порог
    with open(os.path.join(output_dir, 'best_threshold.pkl'), 'wb') as f:
        pickle.dump(best_model['threshold'], f)

    # Сохраняем scaler
    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    # Сохраняем имена признаков
    with open(os.path.join(output_dir, 'feature_names.pkl'), 'wb') as f:
        pickle.dump(feature_names, f)

    # Сохраняем название лучшей модели
    with open(os.path.join(output_dir, 'best_model_name.pkl'), 'wb') as f:
        pickle.dump(best_model_name, f)

    # Сохраняем все модели
    for name, data in results.items():
        model_filename = os.path.join(output_dir, f'{name.lower()}_model.pkl')
        with open(model_filename, 'wb') as f:
            pickle.dump(data['model'], f)

    # Сохраняем важность признаков (если доступна)
    if 'feature_importance' in best_model:
        importance_dict = dict(zip(feature_names, best_model['feature_importance']))
        with open(os.path.join(output_dir, 'feature_importance.pkl'), 'wb') as f:
            pickle.dump(importance_dict, f)

    print(f"\nВсе артефакты сохранены в '{output_dir}'")


def print_feature_importance(
        feature_names: List[str],
        importance_values: np.ndarray,
        model_name: str,
        top_n: int = 10
) -> None:
    """
    Выводит топ-N самых важных признаков.

    Args:
        feature_names: Список имен признаков
        importance_values: Значения важности
        model_name: Название модели (для логирования)
        top_n: Количество выводимых признаков
    """
    importance_dict = dict(zip(feature_names, importance_values))
    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

    print(f"\nТОП-{top_n} ВАЖНЕЙШИХ ПРИЗНАКОВ ({model_name})")
    for name, value in sorted_importance[:top_n]:
        print(f"  {name:25s}: {value:.4f}")


# ОСНОВНАЯ ФУНКЦИЯ

def main():
    """
    Основной пайплайн обучения моделей.

    Порядок выполнения:
    1. Загрузка данных
    2. Предобработка
    3. Обучение всех моделей
    4. Выбор лучшей модели
    5. Сохранение артефактов
    """
    print("ОБУЧЕНИЕ МОДЕЛЕЙ ДЛЯ ПРОГНОЗИРОВАНИЯ ОТТОКА")

    # Инициализация конфигурации
    config = ModelConfig()

    try:
        # Загрузка и предобработка данных
        df = load_data(config.data_file)
        X, y, feature_names = preprocess_data(df)

        # Обучение моделей
        results, scaler = train_models(X, y, config)

        # Выбор лучшей модели
        best_model_name = get_best_model(results)

        # Вывод важности признаков
        best_model = results[best_model_name]
        if 'feature_importance' in best_model:
            print_feature_importance(
                feature_names,
                best_model['feature_importance'],
                best_model_name
            )

        # Сохранение всех артефактов
        save_model_artifacts(
            results,
            scaler,
            feature_names,
            best_model_name,
            config.output_dir
        )

        # Итоговое резюме
        print("ИТОГИ ОБУЧЕНИЯ")
        for name, data in results.items():
            print(f"{name:20s}: F1 = {data['f1']:.4f}, ROC-AUC = {data['roc_auc']:.4f}")
        print(f"\nЛучшая: {best_model_name}")
        print("Обучение успешно завершено!")

    except Exception as e:
        print(f"\nОшибка: {e}")
        raise


if __name__ == "__main__":
    main()