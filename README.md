# 🚦 TrafficVision

A modern web application for detecting traffic lights in images and videos using advanced computer vision and deep learning.

## Features

- 📸 **Image Detection**: Upload images and get real-time traffic light detection
- 🎥 **Video Processing**: Upload videos and get processed videos with detections
- 🎨 **Modern UI**: Beautiful, responsive web interface with drag & drop
- 📊 **Detailed Results**: Shows confidence scores and bounding boxes
- 📥 **Download Results**: Download processed videos directly from the browser
- 🤖 **AI-Powered**: Advanced computer vision powered by deep learning

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Model Path

Make sure your trained model is located at:

```
C:\Users\tusha\OneDrive\Desktop\rg\ultralytics\runs\detect\train10\weights\best.pt
```

If your model is in a different location, update the `model_path` variable in `app.py`.

### 3. Run the Application

```bash
python app.py
```

The web application will start on `http://localhost:5000`

## Usage

1. **Open your browser** and go to `http://localhost:5000`
2. **Upload files** by either:
   - Dragging and dropping files onto the upload area
   - Clicking "Choose File" to browse
3. **Click "Analyze with AI"** to process your file
4. **View results**:
   - For images: See detected traffic lights with confidence scores
   - For videos: Watch the processed video with detections and download it

## Supported File Types

- **Images**: JPG, PNG, WebP
- **Videos**: MP4, AVI, MOV, MKV
- **Maximum file size**: 16MB

## Model Information

- **Classes**: Green, Red, Yellow traffic lights
- **Model**: YOLOv8 trained on TLD-2025-1 dataset
- **Output**: Bounding boxes with confidence scores

## File Structure

```
├── app.py                 # Main Flask application
├── templates/
│   └── index.html        # Web interface
├── uploads/              # Temporary upload folder
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Troubleshooting

### Common Issues

1. **Model not found**: Check if the model path in `app.py` is correct
2. **Import errors**: Make sure all dependencies are installed
3. **Memory issues**: Reduce batch size or use smaller images/videos
4. **CUDA errors**: The app will automatically use CPU if GPU is not available

### Performance Tips

- For faster processing, use smaller images/videos
- Close other applications to free up memory
- Consider using a GPU for better performance

## API Endpoints

- `GET /`: Main web interface
- `POST /predict`: Upload and process files
- `GET /download/<filename>`: Download processed files

## License

This project is for educational and research purposes.
