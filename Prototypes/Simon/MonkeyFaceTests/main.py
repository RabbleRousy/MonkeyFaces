# Train multiple images per person
# Find and recognize faces in an image using a SVC with scikit-learn

import face_recognition
from sklearn import svm
import os

# Training the SVC classifier

# The training data would be all the face encodings from all the known images and the labels are their names
encodings = []
names = []

train_dir_path = '../../../facedata_yamada/mini_set/train/'
# Training directory
train_dir = os.listdir(train_dir_path)

# Loop through each person in the training directory
for person in train_dir:
    if person == '.DS_Store':
        continue

    # Test with human to see how shitty the face detection is
    if person == 'Simon':
        continue

    print("Monkey: " + person)

    training_faces = 0
    count = 0

    pix = os.listdir(train_dir_path + person)

    # Loop through each training image for the current person
    for person_img in pix:
        # Move images you want to exclude from training into a folder named "hidden"
        if person_img == 'hidden':
            continue
        if person_img == '.DS_Store':
            continue
        count += 1
        # Get the face encodings for the face in each image file
        try:
            face = face_recognition.load_image_file(train_dir_path + person + "/" + person_img)
        except Exception as e:
            print("Exception thrown while loading file " + person_img + ":")
            print(e)
            continue
        face_bounding_boxes = face_recognition.face_locations(face)

        # If training image contains exactly one face
        if len(face_bounding_boxes) == 1:
            face_enc = face_recognition.face_encodings(face)[0]
            # Add face encoding for current image with corresponding label (name) to the training data
            encodings.append(face_enc)
            names.append(person)
            training_faces += 1
            print(str(count) + "/" + str(len(pix)) + ' ✅')
        else:
            print(str(count) + "/" + str(len(pix)) + ' ❌')

    print("Extracted " + str(training_faces) + " faces for training from " + str(len(pix)) + " images!")

# Create and train the SVC classifier
clf = svm.SVC(gamma='scale')
clf.fit(encodings, names)

test_dir_path = '../../../facedata_yamada/mini_set/test/'
test_dir = os.listdir(test_dir_path)

for img in test_dir:
    if img == '.DS_Store':
        continue
    # Load the test image with unknown faces into a numpy array
    test_image = face_recognition.load_image_file(test_dir_path + img)

    # Find all the faces in the test image using the default HOG-based model
    face_locations = face_recognition.face_locations(test_image)
    no = len(face_locations)

    # Predict all the faces in the test image using the trained classifier
    for i in range(no):
        test_image_enc = face_recognition.face_encodings(test_image)[i]
        name = str(*clf.predict([test_image_enc]))
        mark = '✅' if img.__contains__(name) else '❌'
        print("Found " + name + " " + mark)
