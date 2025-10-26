from tensorflow.keras import datasets
import matplotlib.pyplot as plt

# Load dataset
(train_images, train_labels), _ = datasets.fashion_mnist.load_data()

# Pick one sample (you can change the index to get another image)
sample_index = 0
img = train_images[sample_index]

# Save it as a 28x28 grayscale PNG
plt.imsave("fashion_sample.png", img, cmap='gray')

print("✅ Saved as fashion_sample.png")
