import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



data = pd.read_csv("models/logs/training_log.csv")

plt.figure(figsize=(12, 6))
#plt.plot(data['epoch'], data['accuracy'], label='train accuracy',color='blue')
plt.plot(data['epoch'], data['val_accuracy'], label='validation accuracy',color='red')
plt.title('trainin accuracy againts validation accuracy')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()