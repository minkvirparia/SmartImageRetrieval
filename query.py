import os
import cv2
import torch
import shutil
import chromadb
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import models
import torchvision.transforms as transforms


class FaceQuery:
    """
    A class to detect faces in an input image, generate embeddings, and retrieve matching images.
    """

    def __init__(self, db_path: str):
        """Initialize ChromaDB client and retrieve collection."""
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name="face_embeddings")
        self.source_dir = "./input"
        self.destination_dir = "./output"
        self.db_path = os.path.join(os.getcwd(), "chroma_db")

        os.makedirs(self.destination_dir, exist_ok=True)

    def extract_face_embedding(self, img_path: str) -> np.ndarray:
        """
        Detect faces and generate embeddings using ResNet.

        :param img_path: Path to input image.
        :return: Face embedding (numpy array) or None if no face is detected.
        """

        image_path = os.path.join(self.source_dir, img_path)
        image_cv = cv2.imread(image_path)

        if image_cv is None:
            print(f"Error reading image: {img_path}")
            return

        # Convert BGR to RGB (OpenCV format to PIL format)
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        # Detect faces
        faces, scores = self.detector.detect(image_pil)

        if faces is None:
            print(f"No faces detected in {img_path}")
            return

        x_min, y_min, x_max, y_max = map(int, faces)

        # Ensure coordinates are within image bounds
        h, w, _ = image_cv.shape
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(w, x_max), min(h, y_max)

        # Extract face region
        face_image = image_cv[y_min:y_max, x_min:x_max]

        # Convert cropped face to PIL image and apply transformations
        face_pil = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
        face_tensor = self.transform(face_pil).unsqueeze(0)  # Add batch dimension

        # Generate embedding using ResNet-50
        with torch.no_grad():
            embedding = self.resnet(face_tensor).squeeze(0).numpy()

        return embedding

    def search_similar_faces(self, query_embedding: np.ndarray, top_k: int = 5):
        """
        Query ChromaDB for similar face embeddings.

        :param query_embedding: NumPy array of the query face embedding.
        :param top_k: Number of closest matches to retrieve.
        :return: List of matching image filenames.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()], n_results=top_k
        )

        if "metadatas" not in results or not results["metadatas"][0]:
            print("No matching faces found.")
            return []

        matching_images = set(meta["image_name"] for meta in results["metadatas"][0])
        return list(matching_images)


def main():
    query_image_path = input("Enter the path of the image to search: ")

    if not os.path.exists(query_image_path):
        print("Error: Image file not found.")
        return

    face_query = FaceQuery()

    print("Extracting face embedding...")
    query_embedding = face_query.extract_face_embedding(query_image_path)

    if query_embedding is None:
        print("No face found in the input image.")
        return

    print("Searching for similar faces...")
    matched_images = face_query.search_similar_faces(query_embedding)

    if not matched_images:
        print("No matching images found.")
        return

    print("Matched images:", matched_images)

    # Copy matching images to output folder
    for img_name in matched_images:
        input_img_path = os.path.join(face_query.source_dir, img_name)
        output_img_path = os.path.join(face_query.destination_dir, img_name)

        if os.path.exists(input_img_path):
            shutil.copy(input_img_path, output_img_path)
            print(f"Copied {img_name} to {face_query.destination_dir}")

    print("Face retrieval completed.")


if __name__ == "__main__":
    main()
