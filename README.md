# SmartImageRetrieval - Reverse Image Search for Face Similarity

## 📌 Introduction
This project is a reverse image search system for face similarity detection. It enables users to extract all photos of a particular person from a large collection, such as wedding albums. By leveraging deep learning-based facial embeddings and an efficient vector search mechanism, the system provides accurate and fast retrieval of matching faces.

## ⚙️ How It Works
1. **Face Detection:** Uses MTCNN to detect faces in images.
2. **Feature Extraction:** Generates high-dimensional face embeddings using ResNet50.
3. **Storage & Indexing:** Stores embeddings in ChromaDB with structured face indexing.
4. **Search & Retrieval:** Queries new face embeddings against the database to return images containing the most similar faces.

## 🛠 Tech Stack
- **Face Detection:** MTCNN
- **Feature Extraction:** ResNet50
- **Vector Database:** ChromaDB

## 🚀 Installation & Setup
```bash
# Clone the repository
https://github.com/minkvirparia/SmartImageRetrieval.git
cd SmartImageRetrieval

# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # For Linux
venv\Scripts\activate     # For Windows

# Install dependencies
pip install -r requirements.txt

# For Detecting faces from all images and generating there embeddings
python face_detection_and_image_embeddings.py

# For storing embeddings into chromdb
python store_face_embeddings.py

# For Quering the new face image
python query.py

```

## 🌟 Features
- **Accurate Face Detection** using MTCNN.
- **Efficient Feature Extraction** with ResNet50.
- **Fast and Scalable Face Search** using ChromaDB.
- **Structured Face Indexing** to link detected faces with their source images.

## 📝 Usage
1. Place all images in the `input/` directory.
2. Run `face_detection_and_image_embeddings.py` to process and generate face embeddings.
3. Run `store_face_embeddings.py` to store generated face embeddings into vector database
4. Query a new face image using `query.py` to retrieve matching images.
5. Resultant images with query face will be stored in `output/` directory
