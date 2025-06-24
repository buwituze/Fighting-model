## Fighting Classification Model

### Running The API:

#### If recreate-model file is neded :

 - `cd Fighting-model`
 - Run: `python app.py`

#### If recreate_model is not needed 

 - `cd Fighting-model`
 - In **app.py**, change this line :
   - `MODEL_PATH = os.path.join(CURRENT_DIR, "fight_detection_model_v2_compatible.h5")`
   to 
   - `MODEL_PATH = os.path.join(CURRENT_DIR, "fight_detection_model_optimized.h5")`
    to targetting the right saved model

 - Run: `python app.py`

### Usage:

**Inputs:**

- Threshold for classification: `0.5` (use this as default for optimal classification)
- Video file: accepts single or multiple files of type `.mp4`, `.avi`, `.mov` 

### API Output Example:

 - "prediction": "noFight",
 - "probability": 0.4763191044330597,
 - "confidence_score": 0.5236808955669403,
 -  "confidence_level": "Low",
 -  "threshold_used": 0.5,
 -  "frames_processed": 16,
 -  "video_duration_estimate": 8.1,
  - "model_status": "original"
