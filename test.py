from colored_list import ColoredList
from numpy.lib.function_base import blackman
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


fe = Face_Embedder()
def face_from_file(filename):
    global fe
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return fe.extract_face(img)


# face_will_1 = face_from_file("will_smith_1.jpg")
face_will_1 = Face_Embedder.extract_face_from_file("will_smith_1.jpg")
face_will_2 = Face_Embedder.extract_face_from_file("will_smith_2.jpg")
face_3 = Face_Embedder.extract_face_from_file("face.jpg")
face_anthony = Face_Embedder.extract_face_from_file("anthony.jpg")

blist = ColoredList()

blist.add_face(face_will_1)
blist.add_face(face_anthony)
blist.add_face(face_3)

ans = blist.search(face_will_2)

# print(ans)