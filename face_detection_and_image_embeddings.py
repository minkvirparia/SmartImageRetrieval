import os
import re
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import models


class FaceEmbeddingExtractor:
    """
    A class to handle face detection using MTCNN and face embedding generation using ResNet-50.
    """

    def __init__(self, source_dir: str, destination_dir: str):
        """
        Initialize MTCNN for face detection and ResNet-50 for embedding extraction.

        :param source_dir: Directory containing images.
        :param destination_dir: Directory to save detected face images.
        """
        self.source_dir = source_dir
        self.destination_dir = destination_dir
        os.makedirs(self.destination_dir, exist_ok=True)

        # Initialize MTCNN detector with optimized thresholds
        self.detector = MTCNN(keep_all=True, thresholds=[0.6, 0.7, 0.98])

        # Load ResNet-50 model (removing the classification layer)
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = (
            torch.nn.Identity()
        )  # Remove classification head to extract embeddings
        self.resnet.eval()  # Set to evaluation mode

        # Define image preprocessing transformations
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),  # ResNet requires 224x224 input
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # Standard normalization
            ]
        )

    def _natural_sort_key(self, filename: str):
        """
        Extracts numerical parts from filenames to perform natural sorting.

        :param filename: File name string.
        :return: List containing alphanumeric parts for sorting.
        """
        return [
            int(text) if text.isdigit() else text.lower()
            for text in re.split(r"(\d+)", filename)
        ]

    def extract_faces_and_embeddings(self):
        """
        Detects faces in images, extracts embeddings, and saves face crops.
        """
        # Read and sort image files
        image_files = sorted(
            [
                img
                for img in os.listdir(self.source_dir)
                if img.lower().endswith(("jpeg", "jpg", "png", "webp"))
            ],
            key=self._natural_sort_key,
        )

        face_data = []  # Store face embeddings

        for image_file in image_files:
            image_path = os.path.join(self.source_dir, image_file)
            image_cv = cv2.imread(image_path)

            if image_cv is None:
                print(f"Error reading image: {image_file}")
                continue

            # Convert BGR to RGB (OpenCV format to PIL format)
            image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)

            # Detect faces
            faces, scores = self.detector.detect(image_pil)

            if faces is None:
                print(f"No faces detected in {image_file}")
                continue

            for idx, face in enumerate(faces):
                x_min, y_min, x_max, y_max = map(int, face)

                # Ensure coordinates are within image bounds
                h, w, _ = image_cv.shape
                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(w, x_max), min(h, y_max)

                # Extract face region
                face_image = image_cv[y_min:y_max, x_min:x_max]

                if face_image.size == 0:  # Skip if face extraction failed
                    continue

                # Save cropped face
                face_filename = f"{os.path.splitext(image_file)[0]}_cropped_{idx+1}.jpg"
                face_path = os.path.join(self.destination_dir, face_filename)
                cv2.imwrite(face_path, face_image)

                # Convert cropped face to PIL image and apply transformations
                face_pil = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
                face_tensor = self.transform(face_pil).unsqueeze(
                    0
                )  # Add batch dimension

                # Generate embedding using ResNet-50
                with torch.no_grad():
                    embedding = self.resnet(face_tensor).squeeze(0).numpy()

                # Store embedding data
                face_data.append(
                    {
                        "file_name": image_file,
                        "face_index": idx,
                        "embedding": embedding.tolist(),
                    }
                )

                print(f"Processed: {face_path}, Embedding Size: {len(embedding)}")

        print("Face extraction and embedding generation completed!")
        return face_data


if __name__ == "__main__":
    source_directory = "./input"
    destination_directory = "./detected_faces"

    face_extractor = FaceEmbeddingExtractor(source_directory, destination_directory)
    face_embeddings = face_extractor.extract_faces_and_embeddings()

    # Save embeddings to a JSON file (optional)
    import json

    with open("face_embeddings.json", "w") as f:
        json.dump(face_embeddings, f, indent=4)
