import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for plots
sns.set(style="whitegrid")

# Sample transport cost data (ZAR = South African Rand)
data = {
    "Mode": ["Minibus Taxi", "Uber Go", "UberX", "Gautrain", "Metrorail", "MetroBus", "Rea Vaya", "Walking"],
    "Avg Cost per Trip (ZAR)": [13, 312/15, 420/15, 92, 13, 9, 10.5, 0],
    "Cost per km (ZAR/km)": [13/15, 312/56, 420/56, 92/36, 13/15, 9/10, 10.5/12, 0],
    "Avg Monthly Cost (ZAR)": [572, 312*22, 420*22, 3254, 581, 352, 420, 0],
    "Avg Travel Time (min)": [63, 50, 48, 38, 107, 84, 75, 0]
}

df = pd.DataFrame(data)

# Plot: Cost per Trip
plt.figure(figsize=(10, 6))
sns.barplot(x="Avg Cost per Trip (ZAR)", y="Mode", data=df, palette="viridis")
plt.title("Average Transport Cost per Trip in Gauteng (Pretoria + Johannesburg)")
plt.xlabel("Cost (ZAR)")
plt.ylabel("Transport Mode")
plt.tight_layout()
plt.show()

# Plot: Monthly Cost Comparison
plt.figure(figsize=(10, 6))
sns.barplot(x="Avg Monthly Cost (ZAR)", y="Mode", data=df, palette="magma")
plt.title("Estimated Monthly Transport Cost by Mode")
plt.xlabel("Monthly Cost (ZAR)")
plt.ylabel("Transport Mode")
plt.tight_layout()
plt.show()

# Plot: Cost per km
plt.figure(figsize=(10, 6))
sns.barplot(x="Cost per km (ZAR/km)", y="Mode", data=df, palette="cubehelix")
plt.title("Cost per Kilometer by Transport Mode")
plt.xlabel("Cost per km (ZAR/km)")
plt.ylabel("Transport Mode")
plt.tight_layout()
plt.show()

# Plot: Travel Time
plt.figure(figsize=(10, 6))
sns.barplot(x="Avg Travel Time (min)", y="Mode", data=df, palette="coolwarm")
plt.title("Average Travel Time by Transport Mode")
plt.xlabel("Travel Time (Minutes)")
plt.ylabel("Transport Mode")
plt.tight_layout()
plt.show()
