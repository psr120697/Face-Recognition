import face_recognition

known_image = face_recognition.load_image_file("obama.jpg")
unknown_image = face_recognition.load_image_file("2.jpg")

face_locations = face_recognition.face_locations(known_image)
face_landmarks_list = face_recognition.face_landmarks(known_image)
biden_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
results = face_recognition.compare_faces([biden_encoding], unknown_encoding)