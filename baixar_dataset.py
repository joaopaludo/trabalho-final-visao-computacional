import kagglehub

# Download latest version
path = kagglehub.dataset_download("lgmoneda/br-coins")

print("Path to dataset files:", path)
