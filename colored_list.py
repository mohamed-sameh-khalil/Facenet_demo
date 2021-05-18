from face import Face
from face_embedder import euclidean


class ColoredList:
    THRESHOLD = 50
    def __init__(self) -> None:
        self.faces = []

    def search(self, face: Face):
        closest = min(self.faces, key=lambda other:euclidean(face, other))

        if closest is None or euclidean(closest, face) > ColoredList.THRESHOLD:
            return False

        return True

    def add_face(self, face: Face):
        self.faces.append(face)

    def remove_face(self, face: Face):
        self.faces.remove(face)

    
list1 = [1,2,3,4,5]

