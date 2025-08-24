import matplotlib.pyplot as plt
import numpy as np

## Bar graphs based on how many devices are owned by each gender

# devices = ('Cars', 'Fans', 'Phones')
# devices_counts = {
#     'Male': np.array([74, 34, 81]),
#     'Female': np.array([73, 41, 58]),
# }
# width = 0.3  # the width of the bars: can also be len(x) sequence


# fig, ax = plt.subplots()
# bottom = np.zeros(3)

# for device, device_count in devices_counts.items():
#     p = ax.bar(devices, device_count, width, label=device, bottom=bottom)
#     bottom += device_count

#     ax.bar_label(p, label_type='center')

# ax.set_title('Number of devices by gender')
# ax.legend()

# plt.show()





labels = ['G1', 'G2', 'G3', 'G4', 'G5']
values = [10, 25, 29, 40, 55]

bars = plt.bar(labels, values, color=['blue'])

for bar in bars:
    bar.set_hatch('/') ## add stripes to bars

plt.legend(title='Groups')
plt.show()