import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score  # Add this import


# функція нормалізації
def min_max_normalization(data):
    # Знаходимо мінімальне та максимальне значення у даних
    min_val = min(data)
    max_val = max(data)

    # Виконуємо мін-макс нормалізацію для кожного значення у даних
    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]

    return normalized_data
# Згенеруємо вхідні ознаки та відповіді у межах [0, 1]
n = 500
x = np.random.rand(n)
y = 2 * x + 1 + np.random.randn(n) * 0.4  # Лінійна регресія з шумом
y = min_max_normalization(y)
df = pd.DataFrame({'x': x, 'y': y})
x = df[['x']].values
# поточний графік для всієї вибірки
plt.figure(figsize=(6, 6))
plt.scatter(x, y, s = 10, color='#000')
plt.grid(True)
plt.show()

# Розділіть дані на навчальну та тестову вибірки (наприклад, 80% навчальної та 20% тестової)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Виведення розмірів навчальної та тестової вибірок
print("Розмір навчальної вибірки:", X_train.shape)
print("Розмір тестової вибірки:", X_test.shape)

# Візуалізація навчальної вибірки
plt.figure(figsize=(6, 6))
plt.scatter(X_train, y_train, s=10, color='blue', label='Навчальна вибірка')
plt.grid(True)

# Візуалізація тестової вибірки
plt.scatter(X_test, y_test, s=10, color='red', label='Тестова вибірка')
plt.grid(True)

plt.xlabel('X-координата')
plt.ylabel('Y-координата')
plt.title('Навчальна та тестова вибірки')
plt.legend()
plt.show()

# Функція для обчислення середнього значення на основі k найближчих сусідів
def custom_knn_regression(x_train, y_train, x_test, k):
    y_pred = []
    for x in x_test:
        # Знаходимо k найближчих сусідів
        nearest_neighbors_indices = np.argsort(np.abs(x_train - x))[:k]
        # Обчислюємо середнє значення відповідей цих сусідів
        y_pred_value = np.mean(y_train[nearest_neighbors_indices])
        y_pred.append(y_pred_value)
    return y_pred

# Згенерувати вхідні ознаки та відповіді у межах [0, 1]
n = 500
x = np.random.rand(n)
y = 2 * x + 1 + np.random.randn(n) * 0.4  # Лінійна регресія з шумом

# Нормалізація відповідей
y = (y - np.min(y)) / (np.max(y) - np.min(y))

# Створити список різних значень k, з якими ви хочете навчити KNN
k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Вивчені моделі KNN для різних значень k
knn_models = {}

# Навчання KNN регресорів з різними значеннями k
for k in k_values:
    y_pred = custom_knn_regression(x, y, x, k)
    knn_models[k] = y_pred

# Підготовка даних для візуалізації
x_test = np.linspace(0, 1, 100)
y_pred = {k: custom_knn_regression(x, y, x_test, k) for k in k_values}

# Візуалізація результатів

for k, y_pred_k in y_pred.items():
  plt.figure(figsize=(10, 6))
  plt.scatter(x, y, s=10, color='blue', label='Дані')
  plt.plot(x_test, y_pred_k, label=f'KNN (k={k})', color='orange')
  plt.grid(True)
  plt.xlabel('X-координата')
  plt.ylabel('Y-координата')
  plt.title('KNN регресори з різними значеннями k')
  plt.legend()
  plt.show()
# Функція для обчислення середнього значення на основі k найближчих сусідів
def custom_knn_regression(x_train, y_train, x_test, k):
    y_pred = []
    for x in x_test:
        # Знаходимо k найближчих сусідів
        nearest_neighbors_indices = np.argsort(np.abs(x_train - x))[:k]
        # Обчислюємо середнє значення відповідей цих сусідів
        y_pred_value = np.mean(y_train[nearest_neighbors_indices])
        y_pred.append(y_pred_value)
    return y_pred

# Згенерувати вхідні ознаки та відповіді у межах [0, 1]
n = 500
x = np.random.rand(n)
y = 2 * x + 1 + np.random.randn(n) * 0.4  # Лінійна регресія з шумом

# Нормалізація відповідей
y = (y - np.min(y)) / (np.max(y) - np.min(y))

# Створити список різних значень k, з якими ви хочете навчити KNN
k_values = [1, 3, 5, 7, 9]

# Словник для зберігання MSE для кожного значення k
mse_values = {}

# Навчання KNN регресорів з різними значеннями k і обчислення MSE
for k in k_values:
    y_pred = custom_knn_regression(x, y, x, k)
    mse = np.mean((y - y_pred) ** 2)
    mse_values[k] = mse

# Вибір найкращого значення k на основі MSE
best_k = min(mse_values, key=mse_values.get)

# Підготовка даних для візуалізації
x_test = np.linspace(0, 1, 100)
y_pred = custom_knn_regression(x, y, x_test, best_k)

# Візуалізація результатів
plt.figure(figsize=(10, 6))
plt.scatter(x, y, s=10, color='blue', label='Дані')
plt.plot(x_test, y_pred, label=f'KNN (k={best_k})')
plt.grid(True)
plt.xlabel('X-координата')
plt.ylabel('Y-координата')
plt.title(f'KNN регресор (k={best_k}) - Найкращий за MSE')
plt.legend()
plt.show()

print(f"Найкраще значення k: {best_k}, MSE: {mse_values[best_k]:.2f}")
