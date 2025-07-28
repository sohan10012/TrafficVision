from flask import Flask, render_template, request, jsonify, send_file
from flask import send_from_directory
import os
import cv2
import numpy as np
import subprocess
import tempfile
import shutil
import json
import re
from PIL import Image
import io
import base64
import uuid
from werkzeug.utils import secure_filename

app = Flask(__name__ , template_folder="../templates",static_folder=None)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20MB max file size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Model path
model_path = r'C:\Users\tusha\OneDrive\Desktop\rg\ultralytics\runs\detect\train10\weights\best.pt'

# Class names for traffic lights
class_names = ['green', 'red', 'yellow']

# Color mapping for traffic lights (BGR format for OpenCV)
traffic_light_colors = {
    'red': (0, 0, 255),      # Red in BGR
    'green': (0, 255, 0),    # Green in BGR  
    'yellow': (0, 255, 255)  # Yellow in BGR
}

def create_custom_yolo_config():
    """Create a custom YOLO configuration with proper color mapping"""
    config_content = f"""
# Custom YOLO configuration for traffic light detection
# Colors for traffic light classes (BGR format)
colors:
  - [0, 255, 0]    # Green
  - [0, 0, 255]    # Red  
  - [0, 255, 255]  # Yellow

# Class names
names:
  - green
  - red
  - yellow

# Visualization settings
line_thickness: 3
labels: True
boxes: True
"""
    return config_content

def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'mp4', 'avi', 'mov', 'mkv'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def run_yolo_prediction(input_path, output_dir=None, save_txt=True):
    """Run YOLO prediction using command line"""
    try:
        # Create temporary output directory if not provided
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
        
        # Check if input file exists
        if not os.path.exists(input_path):
            print(f"Error: Input file does not exist: {input_path}")
            return None, output_dir
        
        # Create custom configuration file for better visualization
        config_file = os.path.join(output_dir, 'custom_config.yaml')
        with open(config_file, 'w') as f:
            f.write(create_custom_yolo_config())
        
        # Run YOLO command with more options for better results
        cmd = [
            'yolo', 'predict',
            f'model={model_path}',
            f'source={input_path}',
            'save=True',
            f'save_txt={str(save_txt).lower()}',
            'save_conf=True',
            'conf=0.05',  # Very low confidence threshold for more detections
            'iou=0.5',   # IoU threshold for NMS
            'show_boxes=True',
            'show_labels=True',
            'line_width=3',
            f'project={output_dir}',
            'name=prediction',
            'exist_ok=True'
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        print(f"YOLO return code: {result.returncode}")
        print(f"YOLO stdout: {result.stdout}")
        print(f"YOLO stderr: {result.stderr}")
        
        if result.returncode != 0:
            print(f"YOLO command failed: {result.stderr}")
            return None, output_dir
        
        return result.stdout, output_dir
        
    except Exception as e:
        print(f"Error running YOLO: {e}")
        import traceback
        traceback.print_exc()
        return None, output_dir

def run_yolo_video_prediction(input_path, output_dir=None, save_txt=True):
    """Run YOLO video prediction using tracking mode for better temporal consistency"""
    try:
        # Create temporary output directory if not provided
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
        
        # Check if input file exists
        if not os.path.exists(input_path):
            print(f"Error: Input file does not exist: {input_path}")
            return None, output_dir
        
        # Create custom configuration file for better visualization
        config_file = os.path.join(output_dir, 'custom_config.yaml')
        with open(config_file, 'w') as f:
            f.write(create_custom_yolo_config())
        
        # Run YOLO command with tracking for videos - use higher confidence and better tracking
        cmd = [
            'yolo', 'track',  # Use track mode for videos
            f'model={model_path}',
            f'source={input_path}',
            'save=True',
            f'save_txt={str(save_txt).lower()}',
            'save_conf=True',
            'conf=0.25',  # Higher confidence threshold to reduce false positives
            'iou=0.7',   # Higher IoU threshold for better NMS
            'show_boxes=True',
            'show_labels=True',
            'line_width=3',
            'tracker=bytetrack.yaml',  # Use ByteTrack for better tracking
            f'project={output_dir}',
            'name=prediction',
            'exist_ok=True'
        ]
        
        print(f"Running video command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        print(f"YOLO return code: {result.returncode}")
        print(f"YOLO stdout: {result.stdout}")
        print(f"YOLO stderr: {result.stderr}")
        
        if result.returncode != 0:
            print(f"YOLO command failed: {result.stderr}")
            return None, output_dir
        
        return result.stdout, output_dir
        
    except Exception as e:
        print(f"Error running YOLO: {e}")
        import traceback
        traceback.print_exc()
        return None, output_dir

def parse_yolo_results(output_dir):
    """Parse YOLO results from the output directory"""
    predictions = []
    summary = {
        'total_detections': 0,
        'class_counts': {'red': 0, 'yellow': 0, 'green': 0},
        'average_confidence': 0.0,
        'frames_with_detections': 0,
        'total_frames': 0
    }
    
    try:
        # Look for labels directory - YOLO creates a subdirectory with the name parameter
        labels_dir = os.path.join(output_dir, 'prediction', 'labels')
        if not os.path.exists(labels_dir):
            # Try alternative directory structure
            labels_dir = os.path.join(output_dir, 'test', 'labels')
        if not os.path.exists(labels_dir):
            # Try looking for any labels directory
            for subdir in os.listdir(output_dir):
                potential_labels_dir = os.path.join(output_dir, subdir, 'labels')
                if os.path.exists(potential_labels_dir):
                    labels_dir = potential_labels_dir
                    break
        
        # If still not found, try the latest predict directory
        if not os.path.exists(labels_dir):
            predict_dirs = [d for d in os.listdir(output_dir) if d.startswith('predict')]
            if predict_dirs:
                latest_predict = sorted(predict_dirs)[-1]  # Get the latest predict directory
                labels_dir = os.path.join(output_dir, latest_predict, 'labels')
                print(f"Using latest predict directory: {latest_predict}")
        
        print(f"Looking for labels in: {labels_dir}")
        if os.path.exists(labels_dir):
            confidences = []
            frame_detections = {}
            
            for label_file in os.listdir(labels_dir):
                if label_file.endswith('.txt'):
                    label_path = os.path.join(labels_dir, label_file)
                    # Extract frame number from filename (e.g., "Result1_10.txt" -> "10")
                    frame_num = label_file.replace('.txt', '')
                    # If frame_num contains underscore, extract the number after it
                    if '_' in frame_num:
                        frame_num = frame_num.split('_')[-1]
                    # Ensure it's a valid number
                    try:
                        int(frame_num)  # Test if it's a valid integer
                    except ValueError:
                        # If not a valid number, use a sequential number
                        frame_num = str(len([f for f in os.listdir(labels_dir) if f.endswith('.txt') and f < label_file]))
                    
                    frame_predictions = []
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 6:  # class, x, y, w, h, conf (tracking may add track_id)
                                class_id = int(parts[0])
                                confidence = float(parts[5])
                                
                                # Parse bounding box coordinates (YOLO format: center_x, center_y, width, height)
                                center_x = float(parts[1])
                                center_y = float(parts[2])
                                width = float(parts[3])
                                height = float(parts[4])
                                
                                # Convert to corner coordinates
                                x1 = center_x - width / 2
                                y1 = center_y - height / 2
                                x2 = center_x + width / 2
                                y2 = center_y + height / 2
                                
                                if class_id < len(class_names):
                                    class_name = class_names[class_id]
                                    pred = {
                                        'class': class_name,
                                        'confidence': confidence,
                                        'frame': frame_num,
                                        'bbox': [x1, y1, x2, y2]
                                    }
                                    
                                    # Add track ID if available (for tracking)
                                    if len(parts) > 6:
                                        pred['track_id'] = int(parts[6])
                                    
                                    predictions.append(pred)
                                    frame_predictions.append(pred)
                                    summary['class_counts'][class_name] += 1
                                    summary['total_detections'] += 1
                                    confidences.append(confidence)
                    
                    if frame_predictions:
                        frame_detections[frame_num] = frame_predictions
                        summary['frames_with_detections'] += 1
                    
                    summary['total_frames'] += 1
            
            if confidences:
                summary['average_confidence'] = sum(confidences) / len(confidences)
                summary['detection_rate'] = summary['frames_with_detections'] / summary['total_frames'] if summary['total_frames'] > 0 else 0
    
    except Exception as e:
        print(f"Error parsing YOLO results: {e}")
        import traceback
        traceback.print_exc()
    
    return predictions, summary

def deduplicate_detections(predictions):
    """Deduplicate detections to count unique traffic lights"""
    if not predictions:
        return predictions, {}
    
    # Group by track ID to identify unique traffic lights
    unique_traffic_lights = {}
    deduplicated_predictions = []
    
    # First, try to use track IDs if available
    tracked_predictions = [p for p in predictions if 'track_id' in p]
    untracked_predictions = [p for p in predictions if 'track_id' not in p]
    
    if tracked_predictions:
        # Use track IDs for deduplication
        for pred in tracked_predictions:
            track_id = pred['track_id']
            
            if track_id not in unique_traffic_lights:
                unique_traffic_lights[track_id] = {
                    'class': pred['class'],
                    'first_frame': pred['frame'],
                    'last_frame': pred['frame'],
                    'max_confidence': pred['confidence'],
                    'detection_count': 1,
                    'predictions': [pred],
                    'method': 'tracking'
                }
                deduplicated_predictions.append(pred)
            else:
                # Update existing track
                track_info = unique_traffic_lights[track_id]
                track_info['last_frame'] = pred['frame']
                track_info['max_confidence'] = max(track_info['max_confidence'], pred['confidence'])
                track_info['detection_count'] += 1
                track_info['predictions'].append(pred)
                deduplicated_predictions.append(pred)
    
    # For untracked predictions, use spatial clustering
    if untracked_predictions:
        print(f"Using spatial clustering for {len(untracked_predictions)} untracked detections")
        clustered_lights = cluster_detections_by_location(untracked_predictions)
        
        for cluster_id, cluster_info in clustered_lights.items():
            unique_traffic_lights[f"cluster_{cluster_id}"] = cluster_info
            deduplicated_predictions.extend(cluster_info['predictions'])
    
    return deduplicated_predictions, unique_traffic_lights

def cluster_detections_by_location(predictions, distance_threshold=50):
    """Cluster detections by spatial proximity to identify unique traffic lights"""
    if not predictions:
        return {}
    
    clusters = {}
    cluster_id = 0
    
    for pred in predictions:
        bbox = pred['bbox']
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        # Find the closest existing cluster
        closest_cluster = None
        min_distance = float('inf')
        
        for cid, cluster in clusters.items():
            cluster_center_x = cluster['center_x']
            cluster_center_y = cluster['center_y']
            
            distance = ((center_x - cluster_center_x) ** 2 + (center_y - cluster_center_y) ** 2) ** 0.5
            
            if distance < distance_threshold and distance < min_distance:
                min_distance = distance
                closest_cluster = cid
        
        if closest_cluster is not None:
            # Add to existing cluster
            cluster = clusters[closest_cluster]
            cluster['predictions'].append(pred)
            cluster['detection_count'] += 1
            cluster['max_confidence'] = max(cluster['max_confidence'], pred['confidence'])
            
            # Update cluster center
            total_x = sum((p['bbox'][0] + p['bbox'][2]) / 2 for p in cluster['predictions'])
            total_y = sum((p['bbox'][1] + p['bbox'][3]) / 2 for p in cluster['predictions'])
            cluster['center_x'] = total_x / len(cluster['predictions'])
            cluster['center_y'] = total_y / len(cluster['predictions'])
            
        else:
            # Create new cluster
            clusters[cluster_id] = {
                'class': pred['class'],
                'center_x': center_x,
                'center_y': center_y,
                'first_frame': pred['frame'],
                'last_frame': pred['frame'],
                'max_confidence': pred['confidence'],
                'detection_count': 1,
                'predictions': [pred],
                'method': 'spatial_clustering'
            }
            cluster_id += 1
    
    return clusters

def interpolate_detections(predictions, total_frames):
    """Interpolate detections across frames for more consistent video detection"""
    if not predictions:
        return predictions
    
    # Group predictions by track ID if available
    track_predictions = {}
    untracked_predictions = []
    
    for pred in predictions:
        if 'track_id' in pred:
            track_id = pred['track_id']
            if track_id not in track_predictions:
                track_predictions[track_id] = []
            track_predictions[track_id].append(pred)
        else:
            untracked_predictions.append(pred)
    
    interpolated_predictions = []
    
    # Interpolate tracked predictions
    for track_id, track_preds in track_predictions.items():
        if len(track_preds) > 1:
            # Sort by frame number with error handling
            try:
                track_preds.sort(key=lambda x: int(x['frame']))
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not sort predictions by frame number: {e}")
                # Skip interpolation for this track if frame numbers are invalid
                interpolated_predictions.extend(track_preds)
                continue
            
            # Interpolate between detected frames
            for i in range(len(track_preds) - 1):
                current_pred = track_preds[i]
                next_pred = track_preds[i + 1]
                
                try:
                    current_frame = int(current_pred['frame'])
                    next_frame = int(next_pred['frame'])
                except (ValueError, TypeError) as e:
                    print(f"Warning: Invalid frame numbers for interpolation: {e}")
                    # Skip interpolation for this pair
                    interpolated_predictions.append(current_pred)
                    continue
                
                # Add current prediction
                interpolated_predictions.append(current_pred)
                
                # Interpolate intermediate frames
                for frame in range(current_frame + 1, next_frame):
                    # Linear interpolation of bounding box
                    alpha = (frame - current_frame) / (next_frame - current_frame)
                    
                    current_bbox = current_pred['bbox']
                    next_bbox = next_pred['bbox']
                    
                    interpolated_bbox = [
                        current_bbox[0] + alpha * (next_bbox[0] - current_bbox[0]),
                        current_bbox[1] + alpha * (next_bbox[1] - current_bbox[1]),
                        current_bbox[2] + alpha * (next_bbox[2] - current_bbox[2]),
                        current_bbox[3] + alpha * (next_bbox[3] - current_bbox[3])
                    ]
                    
                    # Interpolate confidence
                    interpolated_conf = current_pred['confidence'] + alpha * (next_pred['confidence'] - current_pred['confidence'])
                    
                    interpolated_pred = {
                        'class': current_pred['class'],
                        'confidence': interpolated_conf,
                        'frame': str(frame),
                        'bbox': interpolated_bbox,
                        'track_id': track_id,
                        'interpolated': True
                    }
                    
                    interpolated_predictions.append(interpolated_pred)
            
            # Add the last prediction
            interpolated_predictions.append(track_preds[-1])
        else:
            # Single detection, add as is
            interpolated_predictions.extend(track_preds)
    
    # Add untracked predictions
    interpolated_predictions.extend(untracked_predictions)
    
    return interpolated_predictions

def process_image(image_path):
    """Process image and return predictions"""
    try:
        # Run YOLO prediction
        output, output_dir = run_yolo_prediction(image_path)
        
        if output is None:
            return [], ""
        
        # Parse results
        predictions, summary = parse_yolo_results(output_dir)
        
        # Find the predicted image - YOLO creates a subdirectory with the name parameter
        prediction_dir = os.path.join(output_dir, 'prediction')
        if not os.path.exists(prediction_dir):
            # Try alternative directory structure
            prediction_dir = os.path.join(output_dir, 'test')
        if not os.path.exists(prediction_dir):
            # Try looking for any subdirectory with images
            for subdir in os.listdir(output_dir):
                potential_dir = os.path.join(output_dir, subdir)
                if os.path.isdir(potential_dir):
                    for file in os.listdir(potential_dir):
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            prediction_dir = potential_dir
                            break
                    if prediction_dir != os.path.join(output_dir, 'test'):
                        break
        
        if os.path.exists(prediction_dir):
            # Look for the predicted image
            for file in os.listdir(prediction_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    predicted_image_path = os.path.join(prediction_dir, file)
                    break
            else:
                # If no image found, use original
                predicted_image_path = image_path
        else:
            predicted_image_path = image_path
        
        # Convert image to base64
        with open(predicted_image_path, 'rb') as f:
            img_data = f.read()
        
        img_str = base64.b64encode(img_data).decode()
        
        # Clean up
        try:
            shutil.rmtree(output_dir)
        except:
            pass
        
        return predictions, img_str
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return [], ""

def clean_summary_for_json(summary):
    """Clean summary data to make it JSON serializable"""
    if not summary:
        return {}
    
    cleaned_summary = {}
    
    for key, value in summary.items():
        if key == 'track_details':
            # Clean track details - remove complex objects
            cleaned_tracks = {}
            for track_id, track_info in value.items():
                cleaned_tracks[str(track_id)] = {
                    'class': track_info.get('class', ''),
                    'first_frame': str(track_info.get('first_frame', '')),
                    'last_frame': str(track_info.get('last_frame', '')),
                    'max_confidence': float(track_info.get('max_confidence', 0.0)),
                    'detection_count': int(track_info.get('detection_count', 0)),
                    'method': track_info.get('method', 'unknown')
                }
            cleaned_summary[key] = cleaned_tracks
        elif isinstance(value, (int, float, str, bool, list, dict)):
            # These types are JSON serializable
            cleaned_summary[key] = value
        else:
            # Convert other types to string
            cleaned_summary[key] = str(value)
    
    return cleaned_summary

def process_video(video_path):
    """Process video and return predictions with detailed analysis"""
    try:
        print(f"Starting video processing: {video_path}")
        
        # Get video properties first
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return "", "", [], {}
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames, {duration:.2f}s")
        
        # Run YOLO prediction with better error handling
        print("Starting YOLO video prediction...")
        output, output_dir = run_yolo_video_prediction(video_path, save_txt=True)
        
        if output is None:
            print("YOLO prediction failed - trying alternative approach")
            # Try with regular predict instead of track
            output, output_dir = run_yolo_prediction(video_path, save_txt=True)
            
        if output is None:
            print("All YOLO prediction attempts failed")
            return "", "", [], {}
        
        # Parse results
        predictions, summary = parse_yolo_results(output_dir)
        
        # Deduplicate detections to count unique traffic lights
        if predictions:
            predictions, unique_traffic_lights = deduplicate_detections(predictions)
            print(f"After deduplication: {len(unique_traffic_lights)} unique traffic lights found")
            
            # Update summary with unique traffic light information
            unique_class_counts = {'red': 0, 'yellow': 0, 'green': 0}
            for track_id, track_info in unique_traffic_lights.items():
                unique_class_counts[track_info['class']] += 1
            
            summary.update({
                'unique_traffic_lights': len(unique_traffic_lights),
                'unique_class_counts': unique_class_counts,
                'track_details': unique_traffic_lights
            })
        
        # Interpolate detections for more consistent video detection (optional)
        # Disabled by default to avoid redundant detections
        # if predictions:
        #     try:
        #         predictions = interpolate_detections(predictions, total_frames)
        #         print(f"After interpolation: {len(predictions)} total detections")
        #     except Exception as e:
        #         print(f"Warning: Interpolation failed, using raw predictions: {e}")
        #         # Continue with raw predictions if interpolation fails
        
        # Add video properties to summary
        summary.update({
            'total_frames': total_frames,
            'fps': fps,
            'resolution': f"{width}x{height}",
            'duration': duration,
            'detections_per_frame': summary['total_detections'] / total_frames if total_frames > 0 else 0
        })
        
        if 'unique_traffic_lights' in summary:
            print(f"Video analysis complete: {summary['unique_traffic_lights']} unique traffic lights found")
            print(f"Total detections: {summary['total_detections']}")
            print(f"Unique traffic lights by class: {summary['unique_class_counts']}")
        else:
            print(f"Video analysis complete: {summary['total_detections']} detections found")
        
        # Find the predicted video
        prediction_dir = os.path.join(output_dir, 'prediction')
        if not os.path.exists(prediction_dir):
            # Try to find the latest predict directory
            predict_dirs = [d for d in os.listdir(output_dir) if d.startswith('predict')]
            if predict_dirs:
                latest_predict = sorted(predict_dirs)[-1]  # Get the latest predict directory
                prediction_dir = os.path.join(output_dir, latest_predict)
                print(f"Using latest predict directory for video: {latest_predict}")
        
        if os.path.exists(prediction_dir):
            # Look for the predicted video
            for file in os.listdir(prediction_dir):
                if file.lower().endswith(('.mp4', '.avi', '.mov')):
                    predicted_video_path = os.path.join(prediction_dir, file)
                    print(f"Found predicted video: {predicted_video_path}")
                    
                    # Enhance the video with custom colored bounding boxes
                    enhanced_video_path = os.path.join(prediction_dir, 'enhanced_' + file)
                    if enhance_video_visualization(predicted_video_path, enhanced_video_path, predictions):
                        predicted_video_path = enhanced_video_path
                        print(f"Enhanced video created: {enhanced_video_path}")
                    
                    break
            else:
                # If no video found, use original
                predicted_video_path = video_path
                print("No predicted video found, using original")
        else:
            predicted_video_path = video_path
            print("Prediction directory not found, using original video")
        
        # Check if predicted video exists and has content
        if not os.path.exists(predicted_video_path):
            print(f"Predicted video not found: {predicted_video_path}")
            # Return results even if no video was generated (detections might still be found)
            return "", "", predictions, summary
        
        file_size = os.path.getsize(predicted_video_path)
        if file_size == 0:
            print("Predicted video is empty")
            # Return results even if video is empty (detections might still be found)
            return "", "", predictions, summary
        
        print(f"Predicted video size: {file_size} bytes")
        
        # Convert video to base64
        try:
            with open(predicted_video_path, 'rb') as f:
                video_data = f.read()
            
            video_b64 = base64.b64encode(video_data).decode()
            print(f"Video converted to base64: {len(video_b64)} characters")
            
        except Exception as e:
            print(f"Error reading video file: {e}")
            # Return results even if video reading fails (detections might still be found)
            return "", "", predictions, summary
        
        # Clean up
        try:
            shutil.rmtree(output_dir)
        except:
            pass
        
        return video_b64, predicted_video_path, predictions, summary
        
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
        return "", "", [], {}

def enhance_video_visualization(video_path, output_path, predictions):
    """Enhance video visualization with custom colored bounding boxes"""
    try:
        import cv2
        
        # Open the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return False
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get predictions for this frame
            frame_predictions = [p for p in predictions if p.get('frame', '') == str(frame_count)]
            
            # Draw bounding boxes for this frame
            for pred in frame_predictions:
                if 'bbox' in pred:
                    x1, y1, x2, y2 = pred['bbox']
                    class_name = pred['class']
                    confidence = pred['confidence']
                    
                    # Get color for this class
                    color = traffic_light_colors.get(class_name, (255, 255, 255))
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                    
                    # Draw label
                    label = f"{class_name.upper()}: {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    
                    # Draw label background
                    cv2.rectangle(frame, 
                                (int(x1), int(y1) - label_size[1] - 10),
                                (int(x1) + label_size[0], int(y1)),
                                color, -1)
                    
                    # Draw label text
                    cv2.putText(frame, label, 
                              (int(x1), int(y1) - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Write frame
            out.write(frame)
            frame_count += 1
        
        # Release resources
        cap.release()
        out.release()
        
        return True
        
    except Exception as e:
        print(f"Error enhancing video visualization: {e}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        print(f"File uploaded: {filename} ({os.path.getsize(file_path)} bytes)")
        
        # Check if it's a video or image
        file_ext = filename.rsplit('.', 1)[1].lower()
        
        if file_ext in ['mp4', 'avi', 'mov', 'mkv']:
            # Process video
            video_b64, output_path, predictions, summary = process_video(file_path)
            
            # Clean summary for JSON serialization
            cleaned_summary = clean_summary_for_json(summary)
            
            # Always return results if processing completed successfully
            # Even if no detections were found, we still have analysis results
            return jsonify({
                'type': 'video',
                'video_data': video_b64 if video_b64 else "",
                'filename': 'predicted_video.mp4' if video_b64 else "",
                'predictions': predictions,
                'summary': cleaned_summary,
                'has_video': bool(video_b64)
            })
        else:
            # Process image (including WebP)
            predictions, img_str = process_image(file_path)
            
            return jsonify({
                'type': 'image',
                'predictions': predictions,
                'image_data': img_str
            })
    
    except Exception as e:
        print(f"Error in predict endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    
    finally:
        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

@app.route('/download/<filename>')
def download(filename):
    """Download processed video"""
    try:
        return send_file(
            os.path.join(app.config['UPLOAD_FOLDER'], filename),
            as_attachment=True
        )
    except Exception as e:
        return jsonify({'error': f'Error downloading file: {str(e)}'}), 500

@app.route('/test')
def test():
    """Test endpoint to check if YOLO is working"""
    try:
        result = subprocess.run(['yolo', 'help'], capture_output=True, text=True)
        return jsonify({
            'yolo_available': result.returncode == 0,
            'output': result.stdout[:200] if result.stdout else 'No output',
            'error': result.stderr if result.stderr else 'No error'
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/test_video')
def test_video():
    """Test video processing with a sample video"""
    try:
        test_video_path = r"C:\Users\tusha\Downloads\Result1.mp4"
        if os.path.exists(test_video_path):
            print("Testing video processing...")
            video_b64, output_path, predictions, summary = process_video(test_video_path)
            
            # Clean summary for JSON serialization
            cleaned_summary = clean_summary_for_json(summary)
            
            return jsonify({
                'success': True,
                'predictions_count': len(predictions),
                'summary': cleaned_summary,
                'has_video_output': bool(video_b64),
                'video_length': len(video_b64) if video_b64 else 0
            })
        else:
            return jsonify({'error': 'Test video not found'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    uploads_dir = os.path.join(app.root_path, 'uploads')
    return send_from_directory(uploads_dir, filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)