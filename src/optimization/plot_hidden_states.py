import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go


# Load your data
train_features = np.load("/home/mithil/PycharmProjects/lmsys-scoring/data/hidden_states.npy")
train_labels = np.load("/home/mithil/PycharmProjects/lmsys-scoring/data/labels.npy")
valid_features = np.load("/home/mithil/PycharmProjects/lmsys-scoring/data/hidden_states_validation.npy")
valid_labels = np.load("/home/mithil/PycharmProjects/lmsys-scoring/data/labels_validation.npy")
# Option 2: t-SNE
tsne = TSNE(n_components=3, random_state=42)
features_3d = tsne.fit_transform(train_features)

# Create a DataFrame for easier plotting
df = pd.DataFrame({
    'Component 1': features_3d[:, 0],
    'Component 2': features_3d[:, 1],
    'Component 3': features_3d[:, 2],
    'Label': train_labels
})

# Create an interactive 3D scatter plot
fig = px.scatter_3d(df, x='Component 1', y='Component 2', z='Component 3',
                    color='Label', labels={'color': 'Label'},
                    title='3D Visualization of Features')

# Customize the layout
fig.update_layout(scene = dict(
    xaxis_title='Component 1',
    yaxis_title='Component 2',
    zaxis_title='Component 3'),
    width=900, height=700,
    margin=dict(r=20, b=10, l=10, t=40))

# Show the plot
fig.show()

# If you want to save the plot as an interactive HTML file, uncomment the following line:
fig.write_html("3d_visualization.html")

