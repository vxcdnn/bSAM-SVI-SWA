import matplotlib.pyplot as plt
import numpy as np

# Данные для bSAM-SVI-SWA (каждые 10 эпох)
epochs_swa = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
loss_swa = [2.7462, 1.2423, 0.8394, 0.6822, 0.5977, 0.5439, 0.5049, 0.4781, 0.4610, 0.4481, 0.4438]
acc_swa = [20.47, 68.32, 83.24, 87.58, 89.41, 90.86, 91.52, 92.34, 92.50, 92.66, 92.77]

# Данные для bSAM (каждые 5 эпох для наглядности)
epochs_b = list(range(0, 101, 5))
loss_b = [2.347, 1.481, 1.354, 1.363, 1.360, 1.363, 1.352, 1.338, 1.322, 1.292, 1.279, 1.259, 1.231, 1.208, 1.183, 1.166, 1.149, 1.135, 1.130, 1.128, 1.128]
acc_b = [32.37, 56.91, 73.22, 74.04, 74.82, 71.20, 63.34, 74.50, 71.55, 78.58, 78.09, 79.48, 80.50, 83.37, 83.87, 84.17, 85.86, 86.49, 86.55, 86.57, 86.57]

# Создаем график
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# График функции потерь
ax1.plot(epochs_swa, loss_swa, 'o-', linewidth=2, markersize=6, label='bSAM-SVI-SWA', color='#2E86AB')
ax1.plot(epochs_b, loss_b, 's--', linewidth=2, markersize=5, label='bSAM', color='#A23B72')
ax1.set_xlabel('Эпоха', fontsize=11)
ax1.set_ylabel('Loss', fontsize=11)
ax1.set_title('Trainloss', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)
ax1.set_xlim([0, 100])

# График точности
ax2.plot(epochs_swa, acc_swa, 'o-', linewidth=2, markersize=6, label='bSAM-SVI-SWA', color='#2E86AB')
ax2.plot(epochs_b, acc_b, 's--', linewidth=2, markersize=5, label='bSAM', color='#A23B72')
ax2.set_xlabel('Эпоха', fontsize=11)
ax2.set_ylabel('Точность (%)', fontsize=11)
ax2.set_title('Accuracy', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)
ax2.set_xlim([0, 100])
ax2.set_ylim([20, 100])

plt.tight_layout()
plt.show()