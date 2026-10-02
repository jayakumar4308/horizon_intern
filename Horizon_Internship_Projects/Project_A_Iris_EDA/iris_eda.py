import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Iris.csv")

print("\n--- Dataset Shape ---")
print(df.shape)

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Dataset Information ---")
print(df.info())

print("\n--- Summary Statistics ---")
print(df.describe())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Species Distribution ---")
print(df["Species"].value_counts())

print("\n--- Average Measurements by Species ---")
print(df.groupby("Species")[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]].mean())

# Visualization 1: species count
df["Species"].value_counts().plot(kind="bar")
plt.title("Iris Species Distribution")
plt.xlabel("Species")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("species_distribution.png")
plt.close()

# Visualization 2: petal length
for species, group in df.groupby("Species"):
    plt.scatter(group["PetalLengthCm"], group["PetalWidthCm"], label=species)
plt.title("Petal Length vs Petal Width")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.legend()
plt.tight_layout()
plt.savefig("petal_relationship.png")
plt.close()

print("\nEDA completed. Charts saved as PNG files.")
