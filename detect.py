import argparse
import os
import sys
from PIL import Image

# Ensure project root is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.detector import load_model, predict_face
from utils.face_extractor import extract_primary_face
from utils.video_processor import process_video

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}

def main():
    parser = argparse.ArgumentParser(description="Deepfake Detection CLI Tool (Image & Video)")
    parser.add_argument("--input", "-i", required=True, help="Path to input image or video file")
    parser.add_argument("--weights", "-w", default=None, help="Path to custom model weights (.pth)")
    parser.add_argument("--sample-rate", "-s", type=int, default=10, help="Frame sampling rate for video (default: 10)")
    parser.add_argument("--device", "-d", default=None, help="Device to use ('cpu' or 'cuda')")

    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        print(f"Error: File not found at path: {input_path}")
        sys.exit(1)

    ext = os.path.splitext(input_path)[1].lower()

    print(f"==================================================")
    print(f"       DEEPFAKE DETECTION SYSTEM CLI")
    print(f"==================================================")
    print(f"[+] Input File: {input_path}")

    # Load Model
    print(f"[+] Initializing Neural Network Model...")
    model, device = load_model(weights_path=args.weights, device=args.device)
    print(f"[+] Running on device: {device}")

    if ext in IMAGE_EXTENSIONS:
        print(f"[+] Input recognized as IMAGE format.")
        try:
            pil_img = Image.open(input_path).convert('RGB')
            face_crop, box, has_face = extract_primary_face(pil_img, fallback_to_full=True)
            
            if has_face:
                print(f"[+] Face detected at bounding box: {box}")
            else:
                print(f"[!] No distinct face detected. Analyzing full image area.")

            pred = predict_face(model, face_crop, device)

            print("\n---------------- RESULTS ----------------")
            print(f" Prediction  : {pred['label'].upper()}")
            print(f" Confidence  : {pred['confidence']*100:.2f}%")
            print(f" Real Prob   : {pred['probs']['real']*100:.2f}%")
            print(f" Fake Prob   : {pred['probs']['deepfake']*100:.2f}%")
            print("-----------------------------------------\n")

        except Exception as e:
            print(f"Error processing image: {e}")
            sys.exit(1)

    elif ext in VIDEO_EXTENSIONS:
        print(f"[+] Input recognized as VIDEO format.")
        print(f"[+] Sampling 1 frame every {args.sample_rate} frames...")

        try:
            results = process_video(
                video_path=input_path,
                model=model,
                device=device,
                sample_rate=args.sample_rate
            )

            print("\n---------------- RESULTS ----------------")
            print(f" Overall Prediction : {results['overall_label'].upper()}")
            print(f" Overall Confidence : {results['confidence']*100:.2f}%")
            print(f" Avg Fake Prob      : {results['avg_fake_prob']*100:.2f}%")
            print(f" Max Fake Prob      : {results['max_fake_prob']*100:.2f}%")
            print(f" Fake Frame Ratio   : {results['fake_frame_ratio']*100:.1f}% ({int(results['fake_frame_ratio']*results['total_evaluated'])}/{results['total_evaluated']} frames)")
            print(f" Duration           : {results['duration_sec']} seconds ({results['fps']} FPS)")
            print("-----------------------------------------\n")

        except Exception as e:
            print(f"Error processing video: {e}")
            sys.exit(1)
    else:
        print(f"Error: Unsupported file format '{ext}'. Supported formats: {IMAGE_EXTENSIONS | VIDEO_EXTENSIONS}")
        sys.exit(1)

if __name__ == "__main__":
    main()
