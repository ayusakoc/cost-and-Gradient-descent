import numpy as np
import matplotlib.pyplot as plt

# Veri seti (X: Girdi, y: Gerçek çıktı)
X = np.array([1, 2, 3, 4, 5])
y = np.array([1.5, 3.8, 2.5, 4.2, 5.0])

# Model parametreleri (rastgele başlangıç değerleri)
w = 0.0
b = 0.0

# Hiperparametreler
learning_rate = 0.01
iterations = 1000

# Veri seti boyutu
n = len(X)

# Cost değerlerinin kaydedilmesi
cost_history = []

# Gradient descent algoritması
for i in range(iterations):
    # Model tahminleri
    y_pred = w * X + b

    # Cost function (MSE) hesaplama
    cost = (1 / n) * np.sum((y - y_pred) ** 2)
    cost_history.append(cost)

    # Türevlerin hesaplanması
    dw = -(2 / n) * np.sum(X * (y - y_pred))
    db = -(2 / n) * np.sum(y - y_pred)

    # Parametrelerin güncellenmesi
    w = w - learning_rate * dw
    b = b - learning_rate * db

    # Her 100 iterasyonda bir cost değerini yazdıralım
    if i % 100 == 0:
        print(f"Iteration {i}: Cost = {cost:.4f}, w = {w:.4f}, b = {b:.4f}")

# Son parametre değerleri ve cost
print(f"\nFinal: Cost = {cost:.4f}, w = {w:.4f}, b = {b:.4f}")

# Cost değerlerinin grafik gösterimi
plt.figure(figsize=(10, 6))
plt.plot(range(iterations), cost_history, color='b')
plt.xlabel('Iteration')
plt.ylabel('Cost (MSE)')
plt.title('Cost Function Değeri (MSE) Zaman İçerisinde')
plt.grid(True)
plt.show()
