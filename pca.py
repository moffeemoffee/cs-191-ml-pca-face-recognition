import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_lfw_pairs
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from tqdm import tqdm

rnd_state = 191

lfw_pairs_train = fetch_lfw_pairs(subset="train")
lfw_pairs_test = fetch_lfw_pairs(subset="test")
X_train = lfw_pairs_train.data
y_train = lfw_pairs_train.target
X_test = lfw_pairs_test.data
y_test = lfw_pairs_test.target


def cumulative_variance():
    pca = PCA(whiten=True, random_state=rnd_state).fit(X_train)
    cumvar = np.cumsum(pca.explained_variance_ratio_)

    plt.clf()
    fig, ax = plt.subplots()

    ax.set(xlabel='no. of components', ylabel='cumulative explained variance')
    ax.grid()
    ax.plot(cumvar)

    fig.savefig('cumulative-variance.jpg', dpi=300)


def do_pca(n_components=150):
    pca = PCA(n_components=n_components, whiten=True,
              random_state=rnd_state).fit(X_train)

    # Transform the dataset using PCA
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Train SVM
    clf = SVC(gamma='auto', random_state=rnd_state)
    clf = clf.fit(X_train_pca, y_train)

    # Predict
    y_pred = clf.predict(X_test_pca)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, clf


def save_data(plt_x, plt_y, min_components, max_components):
    df = pd.DataFrame({'n_components': plt_x, 'accuracy': plt_y})
    df.set_index('n_components')
    df.to_csv('log.csv', index=False)

    plt.clf()
    fig, ax = plt.subplots()

    ax.set(xlabel='no. of components',
           ylabel='accuracy',
           title='PCA Face Recognition with SVM')
    ax.grid()
    ax.plot(plt_x, plt_y)

    fig.savefig('graph.jpg', dpi=300)


def do_loop_pca(min_components=2, max_components=500):
    plt_x = []
    plt_y = []
    for n_components in tqdm(range(min_components, max_components + 1, 1)):
        accuracy, clf = do_pca(n_components)
        plt_x.append(n_components)
        plt_y.append(accuracy)
        if n_components % 10 == 0:
            save_data(plt_x, plt_y, min_components, max_components)

    # Save final data
    save_data(plt_x, plt_y, min_components, max_components)

    # Show maximum accuracy
    max_accuracy = max(plt_y)
    max_acc_ncomp = plt_x[plt_y.index(max_accuracy)]
    print(
        f'Maximum accuracy: {max_accuracy:.05f} at {max_acc_ncomp} components')
    # 70min-40rz: Maximum accuracy: 0.88509 at 83 components

    # Show final graph
    plt.show()


do_loop_pca()
# cumulative_variance()
