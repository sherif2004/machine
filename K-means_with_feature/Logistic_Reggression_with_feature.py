import os
from PIL import Image
import numpy as np
import pandas as pd
from mlxtend.plotting import plot_decision_regions
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from skimage import io, color, feature, exposure

def extract_hog_features(image):
    # Convert the image to grayscale
    gray_image = color.rgb2gray(image)

    # Calculate HOG features
    hog_features, hog_image = feature.hog(gray_image, visualize=True)

    # Enhance the contrast of the HOG image for better visualization
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    return hog_features, hog_image_rescaled

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

           hog_features, hog_image = extract_hog_features(image)

           # Append HOG features to the features list
           images.append(hog_features)

           # Append the label to the labels list
           labels.append(class_label)


images=np.array(images)
labels=np.array(labels)
#dictionary
path='classes.txt'
file=open(path)
cls={}
for i in file:
 data=i.split('-')
 cls.update({data[0]:data[1]})

X_train,X_test,y_train,y_test=train_test_split(images,labels,test_size=0.2,random_state=4,shuffle=True)
scaler = StandardScaler()
unique_classes = np.unique(labels)
num_classes = len(unique_classes)
y_train_one_hot=np.eye(num_classes,dtype='int')[y_train.astype('int')]
y_test_one_hot=np.eye(num_classes,dtype='int')[y_test.astype('int')]
y_train_flat = np.argmax(y_train_one_hot, axis=1)
y_test_flat = np.argmax(y_test_one_hot, axis=1)

X_train_reduced=np.reshape(X_train, (X_train.shape[0], -1))
X_train_scaled = scaler.fit_transform(X_train_reduced)

X_test_reduced = np.reshape(X_test, (X_test.shape[0], -1))
X_test_scaled = scaler.transform(X_test_reduced)
#import dat from test file
test_path='Testing'
test_df=pd.read_csv('Testing/test.csv',sep=';')
test_images=test_df['Filename'].values
test_labels=test_df['ClassId'].values
images_for_test=[]
for img in test_images:
    image=Image.open(test_path+'\\'+img)
    image=image.resize((30,30))
    image=np.array(image)
    hog_features, hog_image = extract_hog_features(image)
    # Append HOG features to the features list
    images_for_test.append(hog_features)
images_for_test=np.array(images_for_test)
labels_for_test=np.array(test_labels)

labels_for_test_he=np.eye(num_classes,dtype='int')[labels_for_test.astype('int')]
labels_for_test_he_flat = np.argmax(labels_for_test_he, axis=1)###########

images_for_test_reduced=np.reshape(images_for_test, (images_for_test.shape[0], -1))
images_for_test_scaled = scaler.fit_transform(images_for_test_reduced)#########

model=LogisticRegression(multi_class='ovr', max_iter=1000, C=1.0)
model.fit(X_train_reduced,y_train_flat)

y_pred = model.predict(X_test_reduced)
l_n_pred=model.predict(images_for_test_reduced)
print('###############ACCURACY for splited data#################')
accuracy = accuracy_score(y_test_flat,y_pred)
print("Accuracy:", accuracy)
print('###############ACCURACY for test file#################')
accuracy_n_data = accuracy_score(labels_for_test_he_flat,l_n_pred)
print("accuracy_n_data:", accuracy_n_data)
print('###############Confusion Matrix for split#################')
class_labels=['Hump','Give Way','Stop']
cm = confusion_matrix(y_test_flat, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

print('###############Confusion Matrix for test file#################')
class_labels=['Hump','Give Way','Stop']
cm = confusion_matrix(labels_for_test_he_flat, l_n_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

#make prediction on test file
#i=24
#data_one=np.expand_dims(images_for_test[i],axis=0)
#reshaped_image = images_for_test[i].reshape((9, 9))
#plt.imshow(reshaped_image)
#plt.show()

##############roc split###########
y_prob = model.predict_proba(X_test_scaled)

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_one_hot[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(y_test_one_hot.ravel(), y_prob.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure(figsize=(10, 8))

plt.plot(fpr["micro"], tpr["micro"], label=f'Micro-average ROC curve (area = {roc_auc["micro"]:.2f})',
         color='deeppink', linestyle=':', linewidth=4)

for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {cls[str(i)]} (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
##############roc test file###########
y_prob = model.predict_proba(images_for_test_scaled)

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(labels_for_test_he[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(labels_for_test_he.ravel(), y_prob.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure(figsize=(10, 8))

plt.plot(fpr["micro"], tpr["micro"], label=f'Micro-average ROC curve (area = {roc_auc["micro"]:.2f})',
         color='deeppink', linestyle=':', linewidth=4)

for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {cls[str(i)]} (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
