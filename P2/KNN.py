# 0. load data in DataFrame
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn import neighbors
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


target_list = np.array(['no attack', 'attack'])

#Original = 24519 values

df = pd.read_csv('./FullSets/C1K+Attacks.csv') 

# df = pd.read_csv('./FullSets/D8+Attacks.csv') 
# df = pd.read_csv('./task3_dataset.csv') 
# df = df.drop(df.columns[[0, 1, 2, 3]], axis=1)

# df=df.to_numpy()


# df_iris = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
#                      columns= iris['feature_names'] + ['target'])

df.attack = df.attack.astype(int)
# df_iris.target = df_iris.target.astype(int)

# df_iris.head()
print(df.head())

# Params. Correlation
# sns.set()
# sns.heatmap(df.corr(), square=True, annot=True)
# # Magnetic Fields with least relation: MagneticField_z_MEAN | LinearAcceleration_z_MEAN
# plt.show()


# train, test = train_test_split(df_iris[['petal length (cm)','petal width (cm)', 'target']], test_size=0.4)

# Se realiza la separacion de datos previo Shuffle por lo que el modelo cambia cada ejecucion
# Ver mejor alternativa de separacion
# 38 Casos de ataques positivos: Filas 24482-24519

# df=df[::200]
# print(df.shape)

# # Split data into Attacks and no Attacks
# dfNoAttack = df[df['attack'] == 0]
# dfAttack = df[df['attack'] == 1]

# # Get the train/test split for each case
# trainNoAttack, testNoAttack = train_test_split(
#     dfNoAttack[['MagneticField_z_MEAN', 'LinearAcceleration_z_MEAN','Pressure_MEAN', 'attack']], test_size=0.4, shuffle=False)
# trainAttack, testAttack = train_test_split(
#     dfAttack[['MagneticField_z_MEAN', 'LinearAcceleration_z_MEAN','Pressure_MEAN', 'attack']], test_size=0.4, shuffle=False)

# # Prepare data to get merged into final train/test sets
# frameTrain = [trainNoAttack, trainAttack]
# frameTest = [testNoAttack, testAttack]
# train = pd.concat(frameTrain)
# test = pd.concat(frameTest)

# df = df.sample(frac=1) # Shuffle data

#                        Feature   Relevancy
# 0        GyroscopeStat_COV_z_x   0.068270
# 1        GyroscopeStat_COV_z_y   0.073211
# 2         GyroscopeStat_x_MEAN   0.089344
# 3         GyroscopeStat_z_MEAN   0.070876
# 4   LinearAcceleration_COV_z_x   0.060621
# 5   LinearAcceleration_COV_z_y   0.056778
# 6    LinearAcceleration_x_MEAN   0.066431
# 7    LinearAcceleration_z_MEAN   0.097656
# 8        MagneticField_COV_z_x   0.058626
# 9        MagneticField_COV_z_y   0.051260
# 10        MagneticField_x_MEAN   0.084083
# 11        MagneticField_z_MEAN   0.126312
# 12               Pressure_MEAN   0.096533

df.iloc[:,-1]
train, test = train_test_split(df[['GyroscopeStat_x_MEAN','Pressure_MEAN','LinearAcceleration_z_MEAN','GyroscopeStat_z_MEAN','attack']], test_size=0.4, random_state=999)
# train, test = train_test_split(df, test_size=0.4, random_state=999)

print(type(train))
train.reset_index(inplace=True)
test.reset_index(inplace=True)
# print(test.to_string())

# shuffle = False si hay dimensión temporal
cv = KFold(n_splits=27, shuffle=False)

for i, weights in enumerate(['uniform', 'distance']):
    total_scores = []
    for n_neighbors in range(1, 30):
        fold_accuracy = []
        knn = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
        for train_fold, test_fold in cv.split(train):
            # División train test aleatoria
            f_train = train.loc[train_fold]
            f_test = train.loc[test_fold]
            # entrenamiento y ejecución del modelo
            knn.fit(X=f_train.drop(['attack'], axis=1),
                    y=f_train['attack'])
            y_pred = knn.predict(X=f_test.drop(['attack'], axis=1))
            # evaluación del modelo
            acc = accuracy_score(f_test['attack'], y_pred)
            fold_accuracy.append(acc)
        total_scores.append(sum(fold_accuracy)/len(fold_accuracy))

    plt.plot(range(1, len(total_scores)+1), total_scores,
             marker='o', label=weights)
    print('Max Value ' + weights + " : " + str(max(total_scores)) +
          " (" + str(np.argmax(total_scores) + 1) + ")")
    plt.ylabel('Acc')


plt.legend()
plt.show()
print(df.columns.values[:-1])
# constructor
n_neighbors = 4
weights = 'uniform'
knn = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
# fit and predict

# knn.fit(X=train[df.columns.values[:-1]], y=train['attack'])
knn.fit(X=train[['GyroscopeStat_x_MEAN','Pressure_MEAN','LinearAcceleration_z_MEAN','GyroscopeStat_z_MEAN']], y=train['attack'])

# y_pred = knn.predict(X=test[df.columns.values[:-1]])
y_pred = knn.predict(X=test[['GyroscopeStat_x_MEAN','Pressure_MEAN','LinearAcceleration_z_MEAN','GyroscopeStat_z_MEAN']])

acc = accuracy_score(test['attack'], y_pred)
print('Acc', acc)

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
h = .05  # step size in the mesh


# X = train[df.columns.values[:-1]].as_matrix()
X = train[['GyroscopeStat_x_MEAN','Pressure_MEAN','LinearAcceleration_z_MEAN','GyroscopeStat_z_MEAN']].as_matrix()
y = train['attack'].as_matrix()

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].

# Opcion para reducr el tamaño -> Feature Scaling | meshgrid: sparse=True, copy=False
# st = StandardScaler()
# X = st.fit_transform(X)

# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                      np.arange(y_min, y_max, h))
# Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

# # Put the result into a color plot
# Z = Z.reshape(xx.shape)
# plt.figure()
# plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# # Plot also the training points
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
#             edgecolor='k', s=20)
# plt.xlim(xx.min(), xx.max())
# plt.ylim(yy.min(), yy.max())
# plt.title("Classification (k = %i, weights = '%s')"
#           % (n_neighbors, weights))

# plt.show()


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


plot_confusion_matrix(test['attack'], y_pred, classes=target_list, normalize=True,
                      title='Normalized confusion matrix')
plt.show()
