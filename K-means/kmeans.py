import os
from PIL import Image
import numpy as np
from sklearn.metrics import adjusted_rand_score
from collections import Counter
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#import trainning files
root_directory = 'Training'
classes = [class_label for class_label in os.listdir(root_directory)
           if os.path.isdir(os.path.join(root_directory, class_label))][:3]

images = []
labels = []

for class_label in classes:
    class_path = os.path.join(root_directory, class_label)

    if not os.path.isdir(class_path):
        print(f"Skipped non-directory: {class_path}")
        continue

    for image_file in os.listdir(class_path):
        image_path = os.path.join(class_path, image_file)

        if image_file.lower().endswith(('.ppm', '.png', '.jpg', '.jpeg', '.gif')):
           image=Image.open(image_path)
           image=image.resize((30,30))
           image=np.array(image)
           flattened_image = image.flatten()
           normalized_image = flattened_image / 255.0
           images.append(normalized_image)
           labels.append(class_label)


normalized__images=images
labels=np.array(labels)


#import test files
test_images_nor=[]
test_path='Testing'
test_df=pd.read_csv('Testing/test.csv',sep=';')
test_images=test_df['Filename'].values
test_labels=test_df['ClassId'].values
for img in test_images:
    image=Image.open(test_path+'\\'+img)
    image = image.resize((30, 30))
    image = np.array(image)
    flattened_image = image.flatten()
    normalized_image = flattened_image / 255.0
    test_images_nor.append(normalized_image)

labels_test=test_labels




#dictionnary
path='classes.txt'
file=open(path)
cls={}
for i in file:
 data=i.split('-')
 cls.update({data[0]:data[1]})

# model train
X_train=normalized__images
model=KMeans(n_clusters=3,n_init=10,random_state=36)
model.fit(X_train)

# inertia_values = []
# cluster_range = range(1, 11)  # You can adjust the range as needed

# for n_clusters in cluster_range:
#     model = KMeans(n_clusters=n_clusters, n_init=10, random_state=36)
#     model.fit(X_train)
#     inertia_values.append(model.inertia_)
# # Plotting the inertia curve
# plt.plot(cluster_range, inertia_values, marker='o')
# plt.title('KMeans elbows Curve')
# plt.xlabel('Number of Clusters')
# plt.ylabel('Inertia')
# plt.show()

cluster_assignments = model.labels_




#random test
ari = adjusted_rand_score(labels, cluster_assignments)
print("Adjusted Rand Index:", ari)
# print("Cluster Assignments:", cluster_assignments)

# test prediction
test_cluster_assignments = model.predict(test_images_nor)
train_cluster_assignments = model.predict(X_train)

#plotting test set
pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(test_images_nor)

plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=test_cluster_assignments, cmap='viridis', edgecolors='k')
plt.title('KMeans Clustering - Test Set')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
centroids_pca = pca.transform(model.cluster_centers_)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker='X', s=200, c='red', label='Centroids')
plt.show()
test_cluster_assignments = model.predict(test_images_nor)

#plotting train set
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)

plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=train_cluster_assignments, cmap='viridis', edgecolors='k')
for cluster_label in range(len(cls)):
    # Find indices of samples in the current cluster
    cluster_indices = np.where(train_cluster_assignments == cluster_label)[0]

    # Check if the cluster has any samples
    if len(cluster_indices) > 0:
        # Get the true labels for samples in the current cluster
        cluster_labels = labels[cluster_indices]

        # Find the majority class in the cluster
        majority_class = Counter(cluster_labels).most_common(1)[0][0]

        # Annotate the cluster with the majority class name
        plt.text(X_train_pca[cluster_indices, 0].mean(), X_train_pca[cluster_indices, 1].mean(),
                 f'Cluster {cluster_label}\n{majority_class}', fontsize=8, ha='center', va='center')
    else:
        # Annotate the cluster with a message indicating no samples
        plt.text(X_train_pca[cluster_indices, 0].mean(), X_train_pca[cluster_indices, 1].mean(),
                 f'Cluster {cluster_label}\nNo samples', fontsize=8, ha='center', va='center')
plt.title('KMeans Clustering - Training Set')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
centroids_pca = pca.transform(model.cluster_centers_)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker='X', s=200, c='red', label='Centroids')
plt.show()



#roc  for train set
fpr = dict()
tpr = dict()
roc_auc = dict()
unique_clusters = np.unique(train_cluster_assignments)
for cluster_label in unique_clusters:
    # Binary labels: 1 if the sample belongs to the cluster, 0 otherwise
    binary_labels = (train_cluster_assignments == cluster_label).astype(int)

    # One-hot encode the true labels for comparison
    one_hot_labels = np.zeros((len(labels), len(unique_clusters)))
    one_hot_labels[np.arange(len(labels)), train_cluster_assignments] = 1

    # Compute ROC curve
    fpr[cluster_label], tpr[cluster_label], _ = roc_curve(one_hot_labels[:, cluster_label], binary_labels)

    # Compute AUC
    roc_auc[cluster_label] = auc(fpr[cluster_label], tpr[cluster_label])

# Plot ROC curves
plt.figure(figsize=(10, 6))
for cluster_label in unique_clusters:
    plt.plot(fpr[cluster_label], tpr[cluster_label],
             label=f'Cluster {cluster_label} (AUC = {roc_auc[cluster_label]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for KMeans Clusters')
plt.legend(loc='lower right')
plt.show()


# roc for test set
test_cluster_assignments = model.predict(test_images_nor)
fpr = dict()
tpr = dict()
roc_auc = dict()
unique_clusters = np.unique(test_cluster_assignments)

# Assuming labels_test contains the true labels for the test set
labels_test_one_hot = np.eye(len(unique_clusters))[labels_test]

for cluster_label in unique_clusters:
    # Binary labels: 1 if the sample belongs to the cluster, 0 otherwise
    binary_labels = (test_cluster_assignments == cluster_label).astype(int)

    # Compute ROC curve
    fpr[cluster_label], tpr[cluster_label], _ = roc_curve(labels_test_one_hot[:, cluster_label], binary_labels)

    # Compute AUC
    roc_auc[cluster_label] = auc(fpr[cluster_label], tpr[cluster_label])

# Plot ROC curves
plt.figure(figsize=(10, 6))
for cluster_label in unique_clusters:
    plt.plot(fpr[cluster_label], tpr[cluster_label],
             label=f'Cluster {cluster_label} (AUC = {roc_auc[cluster_label]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for KMeans Clusters')
plt.legend(loc='lower right')
plt.show()


#confustion matrix
conf_matrix = confusion_matrix(labels_test, test_cluster_assignments)
class_names = [cls[str(class_label)] for class_label in classes]
# Rest of the plotting code remains the same
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - Training Set')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.show()