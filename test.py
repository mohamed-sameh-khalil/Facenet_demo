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
def embed_from_file(filename):
    global fe
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return fe.extract_and_get_embed(img)


embed_will_1 = embed_from_file("will_smith_1.jpg")
embed_will_2 = embed_from_file("will_smith_2.jpg")
embed_3 = embed_from_file("face.jpg")
embed_anthony = embed_from_file("anthony.jpg")




for dist in [euclidean, cosine]:
    print(dist(embed_will_1, embed_will_2))
    print(dist(embed_will_1, embed_anthony))
    print(dist(embed_will_1, embed_3))
    print(dist(embed_3, embed_will_2))
    print("******************************")