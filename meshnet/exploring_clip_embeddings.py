import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
text_list = ["tshirt left shoulder", "tshirt left collar", "tshirt left armpit", "tshirt left corner",
             "tshirt right shoulder", "tshirt right collar", "tshirt right armpit", "tshirt right corner"]
text_list = ["left shoulder", "left collar", "left armpit", "left corner",
             "right shoulder", "right collar", "right armpit", "right corner"]
text_list = ["left shoulder (ls)", "left collar (lcl)", "left armpit (la)", "left corner (lcr)",
             "right shoulder (rs)", "right collar (rcl)", "right armpit (ra)", "right corner (rcr)"]
text_list = ["ls", "lcl", "la", "lcr",
             "rs", "rcl", "ra", "rcr"]
text = clip.tokenize(text_list).to(device)

with torch.no_grad():
    text_features = model.encode_text(text)
    

# create a plot showing the cosine similarity between each pair of text features
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

text_features = text_features.cpu().numpy()
# transform this into a cosine similarity matrix

text_features_norm = text_features/ np.linalg.norm(text_features, axis=1, keepdims=True)
similarity_matrix = np.dot(text_features_norm, text_features_norm.T)
plt.figure(figsize=(10, 10))
sns.heatmap(similarity_matrix, annot=True, xticklabels=text_list, yticklabels=text_list)
plt.show()

    
print()

