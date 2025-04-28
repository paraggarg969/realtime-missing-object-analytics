# realtime-missing-object-analytics

Real-time video analytics system for detecting missing and new objects using YOLOv8n and DeepSORT.

## Setup
1. Clone the repository: `git clone https://github.com/paraggarg969/realtime-missing-object-analytics`
2. Build Docker image: `docker build -t object-analytics .`
3. Run container: `docker run -v $(pwd)/outputs:/app/outputs object-analytics`
4. input video as use webcam (`video_source=0`).

## Requirements
- Docker
- NVIDIA GPU (optional, for CUDA acceleration)
- Python 3.10 (if running without Docker)

## Outputs
- `missing_objects/*.png
- `Parag_Garg_ML_Intern.docx`: Evaluation report
