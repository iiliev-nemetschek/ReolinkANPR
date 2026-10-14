"""ALPR processing using FastALPR."""

import cv2
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from .logger import logger

try:
    from fast_alpr import ALPR
    FAST_ALPR_AVAILABLE = True
except ImportError:
    FAST_ALPR_AVAILABLE = False
    logger.error("fast-alpr not available! Install with: pip install fast-alpr")


# Stop scanning frames once the same plate is read this many times at this confidence
# and no other plate string has as many reads
EARLY_EXIT_CONFIDENCE = 0.95
EARLY_EXIT_MATCHES = 3


class ALPRProcessor:
    """Process images for license plate recognition using FastALPR."""

    def __init__(self, config):
        self.config = config

        if not FAST_ALPR_AVAILABLE:
            raise ImportError("fast-alpr library not available")

        # Initialize FastALPR with configured models
        self.alpr = ALPR(
            detector_model=config.detector_model,
            ocr_model=config.ocr_model,
            detector_conf_thresh=0.1  # Detection confidence (lower than recognition threshold)
        )

        logger.info(f"FastALPR initialized with {config.detector_model}")

    def new_scan(self) -> "PlateScan":
        """Start an incremental scan: feed frames one by one until it reports a confident plate."""
        return PlateScan(self)

    @staticmethod
    def _ocr_confidence(ocr) -> float:
        """Newer fast-alpr returns per-character confidences; older returns one float."""
        conf = ocr.confidence
        if isinstance(conf, (list, tuple)):
            return float(sum(conf) / len(conf)) if conf else 0.0
        return float(conf)

    def _save_debug_frames(self, frame_bytes_list: List[bytes], save_dir: Path):
        """Save first and last frames for debugging when no plates detected."""
        try:
            debug_dir = save_dir / "debug_frames"
            debug_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Save first frame
            if len(frame_bytes_list) > 0:
                image_array = np.frombuffer(frame_bytes_list[0], np.uint8)
                img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                if img is not None:
                    first_path = debug_dir / f"{timestamp}_first.jpg"
                    cv2.imwrite(str(first_path), img)
                    logger.info(f"Saved first frame: {first_path}")

            # Save last frame
            if len(frame_bytes_list) > 1:
                image_array = np.frombuffer(frame_bytes_list[-1], np.uint8)
                img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                if img is not None:
                    last_path = debug_dir / f"{timestamp}_last.jpg"
                    cv2.imwrite(str(last_path), img)
                    logger.info(f"Saved last frame: {last_path}")

            # Save middle frame
            if len(frame_bytes_list) > 2:
                mid_idx = len(frame_bytes_list) // 2
                image_array = np.frombuffer(frame_bytes_list[mid_idx], np.uint8)
                img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                if img is not None:
                    mid_path = debug_dir / f"{timestamp}_middle.jpg"
                    cv2.imwrite(str(mid_path), img)
                    logger.info(f"Saved middle frame: {mid_path}")

        except Exception as e:
            logger.error(f"Failed to save debug frames: {e}")


class _Candidate:
    """Reads of one plate string accumulated over a scan."""

    def __init__(self, text: str):
        self.text = text
        self.hits = 0
        self.confident_hits = 0
        self.best_confidence = 0.0
        self.best_detection_confidence = 0.0
        self.image = None
        self.crop = None
        self.bbox = None

    def rank(self) -> Tuple[int, float, float]:
        # Majority vote first, then OCR confidence, then detection confidence
        return (self.hits, self.best_confidence, self.best_detection_confidence)


class PlateScan:
    """Incremental ALPR over a stream of frames, with a majority vote across frames."""

    def __init__(self, processor: ALPRProcessor):
        self.processor = processor
        self.min_confidence = processor.config.min_confidence
        self.candidates: Dict[str, _Candidate] = {}
        self.frames_seen = 0
        self._first_frame: Optional[bytes] = None
        self._last_frame: Optional[bytes] = None

    def add_frame(self, frame_bytes: bytes) -> bool:
        """Process one JPEG frame. Returns True once a plate is read consistently enough to stop."""
        self.frames_seen += 1
        n = self.frames_seen
        if self._first_frame is None:
            self._first_frame = frame_bytes
        self._last_frame = frame_bytes

        try:
            img = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                logger.warning(f"Failed to decode frame {n}")
                return False

            for result in self.processor.alpr.predict(img):
                if result.ocr is None:
                    continue
                text = result.ocr.text.upper().replace(' ', '')
                confidence = self.processor._ocr_confidence(result.ocr)

                if confidence < self.min_confidence:
                    logger.debug(f"Frame {n}: Low confidence {confidence:.3f} for {text}")
                    continue

                logger.info(f"Frame {n}: {text} (confidence: {confidence:.3f})")
                self._record(text, confidence, result, img)
        except Exception as e:
            logger.error(f"Error processing frame {n}: {e}")
            return False

        return self._is_consistent()

    def _record(self, text: str, confidence: float, result, img):
        cand = self.candidates.setdefault(text, _Candidate(text))
        cand.hits += 1
        if confidence >= EARLY_EXIT_CONFIDENCE:
            cand.confident_hits += 1

        detection_confidence = result.detection.confidence
        if (confidence, detection_confidence) > (cand.best_confidence, cand.best_detection_confidence):
            cand.best_confidence = confidence
            cand.best_detection_confidence = detection_confidence
            cand.image = img
            bbox = result.detection.bounding_box
            cand.bbox = bbox
            cand.crop = img[int(bbox.y1):int(bbox.y2), int(bbox.x1):int(bbox.x2)].copy()

    def _ranked(self) -> List[_Candidate]:
        return sorted(self.candidates.values(), key=lambda c: c.rank(), reverse=True)

    def _is_consistent(self) -> bool:
        ranked = self._ranked()
        if not ranked:
            return False
        leader = ranked[0]
        runner_up_hits = ranked[1].hits if len(ranked) > 1 else 0
        return leader.confident_hits >= EARLY_EXIT_MATCHES and leader.hits > runner_up_hits

    def finish(self, save_dir: Path) -> Optional[Dict]:
        """Save images for the winning plate and return its details, or None if nothing was read."""
        ranked = self._ranked()
        if not ranked:
            logger.info("No valid plates detected - saving first and last frames for debugging")
            debug_frames = [f for f in (self._first_frame, self._last_frame) if f is not None]
            self.processor._save_debug_frames(debug_frames, save_dir)
            return None

        best = ranked[0]
        if len(ranked) > 1:
            votes = ", ".join(f"{c.text}x{c.hits}" for c in ranked)
            logger.info(f"Plate votes: {votes}")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        images_dir = save_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        image_filename = f"{timestamp}_{best.text}.jpg"
        cv2.imwrite(str(images_dir / image_filename), best.image)

        crop_filename = f"{timestamp}_{best.text}_crop.jpg"
        if best.crop is not None and best.crop.size:
            cv2.imwrite(str(images_dir / crop_filename), best.crop)

        return {
            'plate_number': best.text,
            'confidence': best.best_confidence,
            'image_path': f"images/{image_filename}",
            'plate_crop_path': f"images/{crop_filename}",
            'box_coordinates': {
                'xmin': best.bbox.x1,
                'ymin': best.bbox.y1,
                'xmax': best.bbox.x2,
                'ymax': best.bbox.y2
            },
            'frame_count': self.frames_seen
        }
