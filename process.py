# import libraries here
import json
import pickle

import cv2
import dlib
import numpy as np
from imutils import face_utils
from keras import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


def train_or_load_facial_expression_recognition_model(train_image_paths, train_image_labels):
    """
    Procedura prima listu putanja do fotografija za obucavanje i listu labela za svaku fotografiju iz prethodne liste

    Kada se procedura pozove, ona treba da trenira model ako on nije istraniran, ili da ga samo ucita ako je prethodno istreniran.
    Ako serijalizujete model, serijalizujte ga odmah pored main.py, bez kreiranja dodatnih foldera.
    Napomena: Platforma ne vrsi download serijalizovanih modela i bilo kakvih foldera i sve ce se na njoj ponovo trenirati (vodite racuna o vremenu).
    Serijalizaciju mozete raditi samo da ubrzate razvoj lokalno, da se model ne trenira svaki put.

    Vreme izvrsavanja celog resenja je ograniceno na maksimalno 1h.

    :param train_image_paths: putanje do fotografija za obucavanje
    :param train_image_labels: labele za sve fotografije iz liste putanja za obucavanje
    :return: Objekat modela
    """
    # TODO - Istrenirati model ako vec nije istreniran

    features = []
    features68 = []
    print('HOG started')
    for i, image_path in enumerate(train_image_paths):
        image = load_image(image_path)
        image, points = get_68_points(image)
        image = resize_image(image)

        features.append(get_hog(image).compute(image))
        # features68.append(try_with_68(points))
        print(i)
    print('HOG finished')
    features = np.array(features)

    """serialized = pickle.dumps(features, protocol=0)
    with open("features.dat", "wb") as file:
        file.write(serialized)

    with open("features.dat", "rb") as file:
        s = file.read()
    features = pickle.loads(s)"""

    features = reshape_data(features)
    # features = np.concatenate([features, features68])

    train_image_labels = np.array(train_image_labels)
    """outputs = []
    for label in train_image_labels:
        output = np.zeros(len(classes()))
        output[classes_to_int()[label]] = 1
        outputs.append(output)

    outputs = np.array(outputs)"""

    model = get_svc(features, train_image_labels)
    """param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [10, 1, 0.1, 0.01, 0.001, 'auto', 'scale'], 'kernel': ['rbf', 'poly', 'sigmoid']}
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
    grid.fit(features, train_image_labels)
    print(grid.best_estimator_)"""
    print('Classification finished')
    """ann = load_trained_ann()

    if ann is None:
        print("Traniranje modela zapoceto.")
        ann = get_ann(features, outputs)
        print("Treniranje modela zavrseno.")
        serialize_ann(ann)
    model = ann"""

    return model


def extract_facial_expression_from_image(trained_model, image_path):
    """
    Procedura prima objekat istreniranog modela za prepoznavanje ekspresije lica i putanju do fotografije na kojoj
    se nalazi novo lice sa koga treba prepoznati ekspresiju.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_model: <Model> Istrenirani model za prepoznavanje karaktera
    :param image_path: <String> Putanja do fotografije sa koje treba prepoznati ekspresiju lica
    :return: <String>  Naziv prediktovane klase (moguce vrednosti su: 'anger', 'contempt', 'disgust', 'happiness', 'neutral', 'sadness', 'surprise'
    """

    image = load_image(image_path)
    # display_image(image)
    image, points = get_68_points(image)
    image = resize_image(image)

    features = np.array([get_hog(image).compute(image)])
    features = reshape_data(features)
    # features68 = np.array([try_with_68(points)])
    # features = np.concatenate([features, features68])

    facial_expression = trained_model.predict(features)[0]

    # results = trained_model.predict(np.array(features, np.float32))
    # facial_expression = result(results)
    # TODO - Prepoznati ekspresiju lica i vratiti njen naziv (kao string, iz skupa mogucih vrednosti)

    print(image_path)
    print(facial_expression)
    print(trained_model.predict_proba(features))
    return facial_expression


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)


def display_image(image):
    plt.imshow(image, 'gray')
    plt.show()


def resize_image(image):
    return cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)


def get_hog(img):
    nbins = 9  # broj binova
    cell_size = (32, 32)  # broj piksela po celiji
    block_size = (3, 3)  # broj celija po bloku
    block_stride = (35, 35)

    hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                      img.shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)

    return hog


def reshape_data(input_data):
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx * ny))


def get_svc(features, train_image_labels):
    model = SVC(kernel='linear', probability=True, verbose=True)
    # model = SVC(kernel='poly', probability=True, verbose=True, gamma=1.4, degree=5, C=15)
    """model = SVC(C=100, cache_size=200, class_weight=None, coef0=0.0,
                decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
                max_iter=-1, probability=False, random_state=None, shrinking=True,
                tol=0.001, verbose=False)"""
    model.fit(features, train_image_labels)

    return model


def get_knn(features, train_image_labels):
    model = KNeighborsClassifier(n_neighbors=12)
    model = model.fit(features, train_image_labels)

    return model


def get_ann(features, train_image_labels):
    model = Sequential()
    model.add(Dense(64, input_dim=2268, activation='sigmoid'))
    # model.add(Dense(140, activation='sigmoid'))
    model.add(Dense(80, activation='sigmoid'))
    model.add(Dense(7, activation='softmax'))

    features = np.array(features, np.float32)  # dati ulazi
    train_image_labels = np.array(train_image_labels, np.float32)  # zeljeni izlazi za date ulaze

    # definisanje parametra algoritma za obucavanje
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    # obucavanje neuronske mreze
    model.fit(features, train_image_labels, epochs=1000, batch_size=1, verbose=2, shuffle=True)

    return model


def serialize_ann(ann):
    # serijalizuj arhitekturu neuronske mreze u JSON fajl
    model_json = ann.to_json()
    with open("neuronska.json", "w") as json_file:
        json_file.write(model_json)
    # serijalizuj tezine u HDF5 fajl
    ann.save_weights("neuronska.h5")


def load_trained_ann():
    try:
        # Ucitaj JSON i kreiraj arhitekturu neuronske mreze na osnovu njega
        json_file = open('neuronska.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        ann = model_from_json(loaded_model_json)
        # ucitaj tezine u prethodno kreirani model
        ann.load_weights("neuronska.h5")
        print("Istrenirani model uspesno ucitan.")
        return ann
    except Exception as e:
        # ako ucitavanje nije uspelo, verovatno model prethodno nije serijalizovan pa nema odakle da bude ucitan
        return None


def classes():
    return ['contempt', 'anger', 'disgust', 'neutral', 'happiness', 'surprise', 'sadness']


def classes_to_int():
    return {'contempt': 0, 'anger': 1, 'disgust': 2, 'neutral': 3, 'happiness': 4, 'surprise': 5, 'sadness': 6}


def result(results):
    return classes()[winner(results[0])]


def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]


def get_68_points(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    rect = detector(image, 1)

    shape = predictor(image, rect[0])
    shape = face_utils.shape_to_np(shape)

    (x, y, w, h) = face_utils.rect_to_bb(rect[0])
    return image[y:y + h + 1, x:x + w + 1], shape


def try_with_68(points):
    face_width = abs(points[0][0] - points[16][0])
    face_height = abs(points[19][1] - points[8][1])
    features = []
    features.append(abs(points[21][0] - points[22][0]) / face_width)
    features.append(abs(points[21][0] - points[17][0]) / face_width)
    features.append(abs(points[26][0] - points[22][0]) / face_width)
    features.append(abs(points[36][0] - points[39][0]) / face_width)
    features.append(abs(points[42][0] - points[45][0]) / face_width)
    features.append(abs(points[31][0] - points[35][0]) / face_width)
    features.append(abs(points[48][0] - points[54][0]) / face_width)
    features.append(abs(points[50][0] - points[52][0]) / face_width)
    features.append(abs(points[58][0] - points[56][0]) / face_width)
    features.append(abs(points[0][0] - points[36][0]) / face_width)
    features.append(abs(points[45][0] - points[16][0]) / face_width)
    features.append(abs(points[2][0] - points[31][0]) / face_width)
    features.append(abs(points[35][0] - points[14][0]) / face_width)
    features.append(abs(points[60][0] - points[64][0]) / face_width)
    features.append(abs(points[3][0] - points[48][0]) / face_width)
    features.append(abs(points[54][0] - points[13][0]) / face_width)
    features.append(abs(points[39][0] - points[27][0]) / face_width)
    features.append(abs(points[27][0] - points[42][0]) / face_width)

    features.append(abs(points[19][1] - points[37][1]) / face_height)
    features.append(abs(points[24][1] - points[44][1]) / face_height)
    features.append(abs(points[37][1] - points[41][1]) / face_height)
    features.append(abs(points[43][1] - points[47][1]) / face_height)
    features.append(abs(points[27][1] - points[33][1]) / face_height)
    features.append(abs(points[51][1] - points[57][1]) / face_height)
    features.append(abs(points[49][1] - points[59][1]) / face_height)
    features.append(abs(points[53][1] - points[55][1]) / face_height)
    features.append(abs(points[33][1] - points[51][1]) / face_height)
    features.append(abs(points[62][1] - points[66][1]) / face_height)
    features.append(abs(points[57][1] - points[8][1]) / face_height)
    features.append(abs(points[48][1] - points[6][1]) / face_height)
    features.append(abs(points[54][1] - points[10][1]) / face_height)
    features.append(abs(points[41][1] - points[48][1]) / face_height)
    features.append(abs(points[46][1] - points[54][1]) / face_height)

    return np.array(features)


"""if __name__ == '__main__':
    model = train_or_load_facial_expression_recognition_model(['dataset/train/image-0.jpg', 'dataset/train/image-1.jpg'], ['contempt', 'anger'])
    expression = extract_facial_expression_from_image(model, 'dataset/train/image-0.jpg')
    print(expression)"""
