import face_recognition
import cv2
import numpy as np

# Refers to the webcam
video = cv2.VideoCapture(0)

#Recognize me
my_image = face_recognition.load_image_file("nabameet.jpg")
my_face_encodings = face_recognition.face_encodings(my_image)[0]

#Recognize Bob Dylan
bob_image = face_recognition.load_image_file("bob.jpg")
bob_face_encodings = face_recognition.face_encodings(bob_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    my_face_encodings,
    bob_face_encodings
]
known_face_names = [
    "nabameet",
    "Bob Dylan"
]


face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:

    ret, frame = video.read()

    #for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image to RGB color
    rgb_small_frame = small_frame[:, :, ::-1]

   
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    #results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        #box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)

        #label with name
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        #result
    cv2.imshow('Video', frame)

    # Hit 'x' to exit
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

video.release()
cv2.destroyAllWindows()