import cv2
import numpy as np
from models.detector import predict_face
from utils.face_extractor import extract_primary_face

def process_video(video_path, model, device, sample_rate=10, max_frames=60, progress_callback=None):
    """
    Processes video file frame by frame, runs deepfake detection on extracted faces,
    and calculates aggregated statistics across the video timeline.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration_sec = total_frames / fps if fps > 0 else 0

    if total_frames > 0 and max_frames > 0:
        effective_sample_rate = max(1, total_frames // max_frames)
    else:
        effective_sample_rate = sample_rate

    target_eval_count = min(max_frames, total_frames // effective_sample_rate) if effective_sample_rate > 0 else max_frames

    frame_results = []
    evaluated_count = 0
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % effective_sample_rate == 0:
            face_crop, box, has_face = extract_primary_face(frame, fallback_to_full=True)
            
            # Predict using model
            pred = predict_face(model, face_crop, device)

            timestamp = round(frame_idx / fps, 2)
            frame_info = {
                'frame_idx': frame_idx,
                'timestamp': timestamp,
                'has_face': has_face,
                'box': box,
                'label': pred['label'],
                'is_deepfake': pred['is_deepfake'],
                'confidence': pred['confidence'],
                'fake_prob': pred['probs']['deepfake'],
                'real_prob': pred['probs']['real']
            }

            frame_results.append(frame_info)
            evaluated_count += 1

            if progress_callback:
                progress_callback(evaluated_count, target_eval_count)

            if evaluated_count >= max_frames:
                break

        frame_idx += 1

    cap.release()

    if not frame_results:
        return {
            'overall_label': 'Unknown',
            'confidence': 0.0,
            'avg_fake_prob': 0.0,
            'max_fake_prob': 0.0,
            'fake_frame_ratio': 0.0,
            'total_evaluated': 0,
            'frame_results': []
        }

    # Aggregate scores
    fake_probs = [fr['fake_prob'] for fr in frame_results]
    fake_count = sum(1 for fr in frame_results if fr['is_deepfake'])
    
    avg_fake_prob = float(np.mean(fake_probs))
    max_fake_prob = float(np.max(fake_probs))
    fake_frame_ratio = fake_count / len(frame_results)

    is_overall_fake = (avg_fake_prob > 0.50) or (fake_frame_ratio >= 0.40)
    overall_label = "Deepfake" if is_overall_fake else "Real"
    confidence = avg_fake_prob if is_overall_fake else (1.0 - avg_fake_prob)

    return {
        'overall_label': overall_label,
        'is_deepfake': is_overall_fake,
        'confidence': confidence,
        'avg_fake_prob': avg_fake_prob,
        'max_fake_prob': max_fake_prob,
        'fake_frame_ratio': fake_frame_ratio,
        'total_evaluated': evaluated_count,
        'duration_sec': round(duration_sec, 2),
        'fps': round(fps, 2),
        'frame_results': frame_results
    }
