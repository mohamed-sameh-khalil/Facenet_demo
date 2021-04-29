# function for face detection with mtcnn
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
import numpy as np
from numpy import expand_dims
# TODO to improve performance make this a class to reuse the MTCNN and the model


class Face_Embedder:
    def __init__(self, filename='facenet_keras.h5'):
        self.model = load_model(filename)
        self.detector = MTCNN()

    def extract_and_get_embed(self, img):
        faces = self.extract_face(img)
        if faces is None:
            return None
        return self.get_embed(faces)

    # get the face embedding for one face


    def get_embed(self, face_pixels):
        # scale pixel values
        face_pixels = face_pixels.astype('float32')
        # standardize pixel values across channels (global)
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        # transform face into one sample
        samples = expand_dims(face_pixels, axis=0)

        # make prediction to get embedding, use model.predict for big batches only
        # yhat = self.model.predict(samples)
        yhat = self.model(samples)
        return yhat[0]


    def extract_face_from_file(self, filename, required_size=(160, 160)):
        image = Image.open(filename)
        # convert to RGB, if needed
        image = image.convert('RGB')
        # convert to array
        pixels = asarray(image)
        return self.extract_face(pixels, required_size)


    # extract a single face from a given photograph
    # can be edited to extract all faces
    # returns None if no faces found in image
    # @pixels has to be a numpy array of the image with rgb, it can be of any size
    def extract_face(self, pixels, required_size=(160, 160)):
        # detect faces in the image
        results = self.detector.detect_faces(pixels)
        if not results:
            return None
        # extract the bounding box from the first face
        x1, y1, width, height = results[0]['box']
        # bug fix
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)
        return face_array

def euclidean(embed1, embed2):
    return np.sum(np.square(embed1 - embed2))

def cosine(embed1, embed2):
    return np.sum(embed1 * embed2)




if __name__ == "__main__":
    print("******************************")

