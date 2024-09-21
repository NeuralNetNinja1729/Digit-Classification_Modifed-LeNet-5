import json
import matplotlib.pyplot as plt

history_path = './models/model_history-2.json'
with open(history_path, 'r') as file:
    history_data = json.load(file)

epoch_list = list(range(1, len(history_data['loss']) + 1))

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(epoch_list, history_data['loss'], 'o-', label='Training Loss')
ax.plot(epoch_list, history_data['val_loss'], 'o-', label='Validation Loss')

ax.set_title('Training vs Validation Loss', fontsize=14)
ax.set_xlabel('Epoch Number', fontsize=12)
ax.set_ylabel('Loss Value', fontsize=12)
ax.set_xticks(epoch_list)

ax.grid(True)
ax.legend()
plt.show()
