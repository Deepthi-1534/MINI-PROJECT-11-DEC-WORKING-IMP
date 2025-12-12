Camouflage Detection and Species Identification System

This project implements an integrated system for detecting camouflaged animals, segmenting their outline, identifying species, and generating analytical metrics. The solution combines YOLO-based object detection, SINet camouflage segmentation, and the Google Gemini multimodal API for species classification. A React frontend provides a clean visual interface for reviewing detection results.

Overview

The application performs the following operations:

Detects animals in images using YOLOv8.

Performs camouflage segmentation using a pre-trained SINet-V2 model (COD10K dataset).

Computes camouflage percentage and overall detection confidence.

Identifies the species via the Gemini API, returning both common and scientific names.

Renders bounding boxes, segmentation maps, and statistics in a modern frontend interface.

System Architecture

Backend

Python 3.10

FastAPI

Ultralytics YOLOv8

SINet-V2 segmentation

Google Gemini multimodal API

NumPy, Torch, OpenCV

Frontend

React with TypeScript

Vite

Tailwind CSS

ShadCN UI components

Project Structure
backend/
 ├── app.py
 ├── models/
 │    ├── detect_infer.py
 │    ├── seg_infer.py
 │    ├── classify_infer.py
 │    ├── sinet_model.py
 │    └── sinet_loader.py
 ├── utils/
 │    ├── camo_utils.py
 │    ├── io_utils.py
 ├── requirements.txt

frontend/
 ├── src/
 ├── public/
 ├── package.json

Backend Setup

Create and activate a virtual environment:

python -m venv venv
venv\Scripts\activate


Install dependencies:

pip install -r backend/requirements.txt


Create a .env file inside the backend directory:

GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=models/gemini-2.5-flash


Start the backend:

cd backend
uvicorn app:app --reload


Backend URL:
http://localhost:8080

Frontend Setup

Install dependencies:

cd frontend
npm install


Start the development server:

npm run dev


Frontend URL:
http://localhost:5173

Processing Workflow

YOLOv8 performs object detection and filters out low-confidence predictions.

SINet-V2 performs camouflage segmentation, producing a mask and probability map.

A detected crop is sent to Gemini for species identification, which returns:

Common name

Scientific name

Confidence score

Camouflage percentage is computed using mask density and region analysis.

The frontend displays detection results, overlays, metrics, and descriptive information.

API Response Format

The backend returns a structured JSON response:

{
  "detected": true,
  "species": "Eastern gray squirrel",
  "scientificName": "Sciurus carolinensis",
  "camouflagePercentage": 64,
  "confidence": 92,
  "boundingBox": { ... },
  "camouflageRegions": [ ... ],
  "description": "...",
  "adaptations": [ ... ]
}

Customization Options

YOLO model selection and confidence thresholds

Segmentation sensitivity parameters

Gemini model version

Bounding box selection logic

Frontend visualization components

Camouflage scoring algorithms

License

This project is intended for academic and research use. Users must comply with licensing requirements for YOLO, SINet, and the Gemini API.