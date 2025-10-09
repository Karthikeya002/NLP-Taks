import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
np.random.seed(42)
data = pd.DataFrame({
'Category': ['A', 'B', 'C', 'D', 'E'],
'Values': np.random.randint(10, 100, 5)
})

# --- Matplotlib Plot ---
plt.figure(figsize=(6, 4))
plt.bar(data['Category'], data['Values'], color='blue')
plt.xlabel('Category')
plt.ylabel('Values')
plt.title('Bar Chart using Matplotlib')
# Save the figure instead of showing it
plt.savefig('matplotlib_bar_chart.png') 

# --- Seaborn Plot (with warning fix) ---
sns.set(style="whitegrid")
plt.figure(figsize=(6, 4))
# Fixed the FutureWarning by adding hue and legend=False
sns.barplot(x='Category', y='Values', data=data, hue='Category', palette='viridis', legend=False)
plt.title('Bar Chart using Seaborn')
# Save the figure instead of showing it
plt.savefig('seaborn_bar_chart.png')