from colored_list import ColoredList
import numpy as np
from face_embedder import *
import time
# from keras.models import load_model
import cv2
from PIL import Image


def time(fun):
    print("*********************************")
    start_time = time.time()
    fun()
    print("--- %s seconds ---" % (time.time() - start_time))
    print("*********************************")


def face_from_file(filename):
    img = cv2.imread(filename)
    return Face_Embedder.extract_face_from_mat(img)


# face_will_1 = face_from_file("will_smith_1.jpg")
face_will_1 = face_from_file("will_smith_1.jpg")
face_will_2 = Face_Embedder.extract_face_from_file("will_smith_2.jpg")
face_will_3 = Face_Embedder.extract_face_from_file("will_smith_3.jpg")
face_3 = Face_Embedder.extract_face_from_file("face.jpg")
face_anthony = Face_Embedder.extract_face_from_file("anthony.jpg")

blist = ColoredList()

blist.add_face(face_will_1)
blist.add_face(face_anthony)
blist.add_face(face_3)


# print("**************************************************************")
print(blist.search(face_will_2))
print(blist.search(face_will_3))
print(blist.search(face_3))
# face2 = Face_Embedder.extract_face(fun2("will_smith_1.jpg"))
# print(face1.embed)
# print(face2.embed)
# print(np.array_equal(face2.embed, face1.embed))


# print(ans)
