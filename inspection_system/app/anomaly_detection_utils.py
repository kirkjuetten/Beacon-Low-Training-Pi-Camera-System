from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import cv2
import pickle
from pathlib import Path


def compute_ssim(image1: np.ndarray, image2: np.ndarray) -> float:
    """Compute Structural Similarity Index between two images."""
    if image1.shape != image2.shape:
        # Resize to match if needed, but ideally they should be aligned
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    return ssim(image1, image2, multichannel=True, data_range=255)


def compute_histogram_similarity(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """Compute histogram intersection similarity."""
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)


def extract_features(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Extract features for anomaly detection."""
    # Color histograms
    hist_b = cv2.calcHist([image], [0], mask, [256], [0, 256]).flatten()
    hist_g = cv2.calcHist([image], [1], mask, [256], [0, 256]).flatten()
    hist_r = cv2.calcHist([image], [2], mask, [256], [0, 256]).flatten()

    # Texture features (simple: mean, std of gradients)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    grad_mean = np.mean(grad_mag[mask > 0])
    grad_std = np.std(grad_mag[mask > 0])

    # Shape features (moments)
    moments = cv2.moments(mask.astype(np.uint8))
    hu_moments = cv2.HuMoments(moments).flatten()

    features = np.concatenate([hist_b, hist_g, hist_r, [grad_mean, grad_std], hu_moments])
    return features


class AnomalyDetector:
    def __init__(self, model_path: Path = None):
        self.scaler = StandardScaler()
        self.ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
        self.model_path = model_path or Path("anomaly_model.pkl")

    def train(self, features_list: list[np.ndarray]):
        """Train on features from good samples."""
        X = np.array(features_list)
        X_scaled = self.scaler.fit_transform(X)
        self.ocsvm.fit(X_scaled)

    def predict(self, features: np.ndarray) -> float:
        """Return anomaly score (negative = anomaly)."""
        X_scaled = self.scaler.transform([features])
        return self.ocsvm.decision_function(X_scaled)[0]

    def save_model(self):
        with open(self.model_path, 'wb') as f:
            pickle.dump({'scaler': self.scaler, 'ocsvm': self.ocsvm}, f)

    def load_model(self):
        if self.model_path.exists():
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                self.scaler = data['scaler']
                self.ocsvm = data['ocsvm']


def detect_anomalies(
    sample_image: np.ndarray,
    reference_image: np.ndarray,
    sample_mask: np.ndarray,
    detector: AnomalyDetector = None
) -> dict:
    """Compute anomaly metrics."""
    metrics = {}

    # SSIM for overall similarity
    metrics['ssim'] = compute_ssim(sample_image, reference_image)

    # Histogram similarity
    hist_ref = cv2.calcHist([reference_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist_sample = cv2.calcHist([sample_image], [0, 1, 2], sample_mask.astype(np.uint8), [8, 8, 8], [0, 256, 0, 256, 0, 256])
    metrics['histogram_similarity'] = compute_histogram_similarity(hist_ref, hist_sample)

    # MSE
    diff = cv2.absdiff(sample_image, reference_image)
    metrics['mse'] = np.mean(diff**2)

    # If detector is trained, use ML score
    if detector:
        features = extract_features(sample_image, sample_mask)
        metrics['anomaly_score'] = detector.predict(features)

    return metrics</content>
<parameter name="filePath">c:\Users\kjuetten\Documents\GitHub\beacon-ai-inspection-camera\inspection_system\app\anomaly_detection_utils.py