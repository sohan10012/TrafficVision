import sys
import os

# Add the current directory to Python path
sys.path.append(os.getcwd())

# Import the video processing function
from TrafficVision.api.index import process_video

def test_video_processing():
    """Test video processing directly"""
    
    test_video_path = r"C:\Users\tusha\Downloads\Result1.mp4"
    
    if not os.path.exists(test_video_path):
        print(f"Test video not found: {test_video_path}")
        return
    
    print(f"Testing video processing with: {test_video_path}")
    print("This may take a few minutes...")
    
    try:
        # Call the video processing function directly
        video_b64, output_path, predictions, summary = process_video(test_video_path)
        
        print("\n=== Video Processing Results ===")
        print(f"Video output path: {output_path}")
        print(f"Number of predictions: {len(predictions)}")
        print(f"Has video data: {bool(video_b64)}")
        print(f"Video data length: {len(video_b64) if video_b64 else 0}")
        
        if summary:
            print(f"\n=== Summary ===")
            print(f"Total detections: {summary.get('total_detections', 0)}")
            print(f"Class counts: {summary.get('class_counts', {})}")
            print(f"Average confidence: {summary.get('average_confidence', 0):.3f}")
            print(f"Total frames: {summary.get('total_frames', 0)}")
            print(f"Frames with detections: {summary.get('frames_with_detections', 0)}")
        
        if predictions:
            print(f"\n=== Sample Predictions ===")
            for i, pred in enumerate(predictions[:5]):  # Show first 5 predictions
                print(f"Prediction {i+1}: {pred.get('class', 'unknown')} - "
                      f"Confidence: {pred.get('confidence', 0):.3f} - "
                      f"Frame: {pred.get('frame', 'unknown')}")
        
        if video_b64:
            print(f"\n✅ Video processing successful! Video data generated.")
        else:
            print(f"\n⚠️ Video processing completed but no video output generated.")
            print("This might be normal if no detections were found or if there were processing issues.")
            
    except Exception as e:
        print(f"❌ Error during video processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_video_processing() 