import os
import json
import chromadb
import numpy as np


class FaceEmbeddingStorage:
    """
    A class to handle storing and querying face embeddings using ChromaDB.
    """

    def __init__(self, db_path: str):
        """
        Initialize ChromaDB client and create a collection.

        :param db_path: Path where ChromaDB database will be stored.
        """

        try:
            self.client = chromadb.PersistentClient(path=db_path)

            self.collection = self.client.get_or_create_collection(
                name="face_embeddings"
            )
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            exit(1)

    def store_embeddings(self, embeddings_file: str):
        """
        Load embeddings from a JSON file and store them in ChromaDB.

        :param embeddings_file: JSON file containing face embeddings.
        """

        if not os.path.exists(embeddings_file):
            print(f"Error: {embeddings_file} not found.")
            return

        # Load embeddings from JSON
        with open(embeddings_file, "r") as f:
            face_data = json.load(f)

        if not face_data:
            print("Error: No face data found in JSON.")
            return

        for face in face_data:
            image_name = face["file_name"]  # Original image filename
            face_id = (
                f"{image_name}_Face_{face['face_index']}"  # Unique face identifier
            )
            embedding = np.array(face["embedding"]).tolist()  # Convert back to list

            # Add to ChromaDB
            self.collection.add(
                ids=[face_id],
                embeddings=[embedding],
                metadatas=[{"image_name": image_name, "face_id": face_id}],
            )

            count += 1

        print("Face embeddings successfully stored in ChromaDB!")

    def query_similar_faces(self, query_embedding: np.ndarray, top_k: int = 5):
        """
        Query ChromaDB for the most similar faces.

        :param query_embedding: NumPy array of the query face embedding.
        :param top_k: Number of closest matches to retrieve.
        :return: List of unique matching image filenames.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()], n_results=top_k
        )

        if "metadatas" not in results or not results["metadatas"][0]:
            print("No matching faces found.")
            return []

        # Extract only unique image names
        matching_images = set(meta["image_name"] for meta in results["metadatas"][0])
        return list(matching_images)


if __name__ == "__main__":
    # db_directory = os.getcwd()  # Store database in current working directory
    # db_path = os.path.join(db_directory, "chroma_db")
    db_path = "./chroma_db"

    face_storage = FaceEmbeddingStorage(db_path=db_path)

    # Store embeddings from the previous script
    embeddings_file = "face_embeddings.json"
    face_storage.store_embeddings(embeddings_file)