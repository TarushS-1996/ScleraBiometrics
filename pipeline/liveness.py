import math
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from pipeline.segmentation import seg_model


class EyeLivenessDetector:
	"""
	Liveness detector for eye images.

	Current features:
	1) Pupil detection from segmentation-first pipeline (implemented)
	2) Vein-flow based liveness (scaffold only)
	"""

	def __init__(
		self,
		model: Any = seg_model,
		seg_size: Tuple[int, int] = (128, 128),
		full_size: Tuple[int, int] = (512, 512),
	) -> None:
		self.model = model
		self.seg_size = seg_size
		self.full_size = full_size

	def _segment_first(self, image_bgr: np.ndarray) -> Dict[str, np.ndarray]:
		"""Step 1: pass image through segmentation model."""
		if image_bgr is None or not isinstance(image_bgr, np.ndarray):
			raise ValueError("image_bgr must be a valid numpy image array")
		if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
			raise ValueError("image_bgr must be an HxWx3 BGR image")

		img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
		img_full = cv2.resize(img_rgb, self.full_size, interpolation=cv2.INTER_AREA)
		img_small = cv2.resize(img_rgb, self.seg_size, interpolation=cv2.INTER_AREA)

		inp = np.expand_dims(img_small.astype(np.float32) / 255.0, axis=0)
		logits_small = self.model.predict(inp, verbose=0)[0]
		class_small = np.argmax(logits_small, axis=-1).astype(np.uint8)
		class_full = cv2.resize(class_small, self.full_size, interpolation=cv2.INTER_NEAREST)

		return {
			"image_rgb": img_full,
			"class_mask": class_full,
			"logits_small": logits_small,
		}

	@staticmethod
	def _largest_component(binary_mask: np.ndarray) -> np.ndarray:
		if binary_mask.max() == 0:
			return binary_mask

		contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		if not contours:
			return np.zeros_like(binary_mask)

		largest = max(contours, key=cv2.contourArea)
		out = np.zeros_like(binary_mask)
		cv2.drawContours(out, [largest], -1, 255, thickness=cv2.FILLED)
		return out

	@staticmethod
	def _shape_stats(mask: np.ndarray) -> Dict[str, float]:
		contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		if not contours:
			return {"circularity": 0.0, "cx": 0.0, "cy": 0.0, "bbox_x": 0.0, "bbox_y": 0.0, "bbox_w": 0.0, "bbox_h": 0.0}

		c = max(contours, key=cv2.contourArea)
		area = float(cv2.contourArea(c))
		perimeter = float(cv2.arcLength(c, True))
		circularity = 0.0
		if perimeter > 0:
			circularity = float((4.0 * math.pi * area) / (perimeter * perimeter))

		moments = cv2.moments(c)
		cx, cy = 0.0, 0.0
		if moments["m00"] > 0:
			cx = moments["m10"] / moments["m00"]
			cy = moments["m01"] / moments["m00"]

		x, y, w, h = cv2.boundingRect(c)
		return {
			"circularity": max(0.0, min(1.0, circularity)),
			"cx": float(cx),
			"cy": float(cy),
			"bbox_x": float(x),
			"bbox_y": float(y),
			"bbox_w": float(w),
			"bbox_h": float(h),
		}

	def _choose_pupil_class(self, image_rgb: np.ndarray, class_mask: np.ndarray) -> Tuple[Optional[int], float, np.ndarray]:
		"""
		Try to identify pupil class from segmentation output.
		This is robust to unknown class ids by scoring candidate classes.
		"""
		h, w = class_mask.shape
		gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

		unique_ids = [int(v) for v in np.unique(class_mask)]
		candidates = [cid for cid in unique_ids if cid != 1]

		best_id: Optional[int] = None
		best_score = -1.0
		best_mask = np.zeros_like(class_mask, dtype=np.uint8)

		image_diag = math.sqrt(float(h * h + w * w)) + 1e-6
		min_area = int(0.0005 * h * w)
		max_area = int(0.20 * h * w)

		for cid in candidates:
			raw = (class_mask == cid).astype(np.uint8) * 255
			raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
			raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
			cand = self._largest_component(raw)

			area = int(np.count_nonzero(cand))
			if area < min_area or area > max_area:
				continue

			pix = gray[cand > 0]
			if pix.size == 0:
				continue

			mean_intensity = float(np.mean(pix))
			darkness_score = 1.0 - (mean_intensity / 255.0)

			stats = self._shape_stats(cand)
			cx = stats["cx"]
			cy = stats["cy"]
			center_dist = math.sqrt((cx - (w / 2.0)) ** 2 + (cy - (h / 2.0)) ** 2) / image_diag
			center_score = max(0.0, 1.0 - (center_dist / 0.40))

			area_ratio = area / float(h * w)
			target_area_ratio = 0.03
			size_score = max(0.0, 1.0 - abs(area_ratio - target_area_ratio) / target_area_ratio)

			circularity = stats["circularity"]
			score = (0.45 * darkness_score) + (0.25 * center_score) + (0.15 * circularity) + (0.15 * size_score)

			if score > best_score:
				best_score = score
				best_id = cid
				best_mask = cand

		return best_id, max(best_score, 0.0), best_mask

	def _fallback_dark_pupil(self, image_rgb: np.ndarray, class_mask: np.ndarray) -> np.ndarray:
		"""
		Fallback when segmentation does not expose a clear pupil class.
		Still segmentation-first: eye region from model constrains dark-region search.
		"""
		h, w = class_mask.shape
		gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
		blur = cv2.GaussianBlur(gray, (7, 7), 0)

		_, dark = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

		eye_region = (class_mask != 0).astype(np.uint8) * 255
		if np.count_nonzero(eye_region) == 0:
			eye_region[:, :] = 255

		ellipse_mask = np.zeros((h, w), dtype=np.uint8)
		cv2.ellipse(
			ellipse_mask,
			center=(w // 2, h // 2),
			axes=(int(w * 0.28), int(h * 0.22)),
			angle=0,
			startAngle=0,
			endAngle=360,
			color=255,
			thickness=-1,
		)

		cand = cv2.bitwise_and(dark, dark, mask=eye_region)
		cand = cv2.bitwise_and(cand, cand, mask=ellipse_mask)
		cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
		cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
		cand = self._largest_component(cand)
		return cand

	@staticmethod
	def _compute_pupil_metrics(pupil_mask: np.ndarray) -> Dict[str, Any]:
		area_px = int(np.count_nonzero(pupil_mask))
		if area_px <= 0:
			return {
				"area_px": 0,
				"diameter_px": 0.0,
				"center": None,
				"bbox": None,
			}

		stats = EyeLivenessDetector._shape_stats(pupil_mask)
		diameter_px = float(math.sqrt((4.0 * area_px) / math.pi))
		return {
			"area_px": area_px,
			"diameter_px": diameter_px,
			"center": (stats["cx"], stats["cy"]),
			"bbox": (
				int(stats["bbox_x"]),
				int(stats["bbox_y"]),
				int(stats["bbox_w"]),
				int(stats["bbox_h"]),
			),
		}

	@staticmethod
	def _overlay_pupil(image_rgb: np.ndarray, pupil_mask: np.ndarray) -> np.ndarray:
		overlay = image_rgb.copy()
		contours, _ = cv2.findContours(pupil_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		if contours:
			cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
		return cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

	def detect_pupil(self, image_bgr: np.ndarray) -> Dict[str, Any]:
		"""
		Step 2: liveness process using segmentation output to detect pupil.
		"""
		segmented = self._segment_first(image_bgr)
		image_rgb = segmented["image_rgb"]
		class_mask = segmented["class_mask"]

		class_id, confidence, pupil_mask = self._choose_pupil_class(image_rgb, class_mask)
		used_fallback = False
		if np.count_nonzero(pupil_mask) == 0:
			pupil_mask = self._fallback_dark_pupil(image_rgb, class_mask)
			used_fallback = True
			confidence = 0.35 if np.count_nonzero(pupil_mask) > 0 else 0.0

		metrics = self._compute_pupil_metrics(pupil_mask)
		overlay_bgr = self._overlay_pupil(image_rgb, pupil_mask)

		return {
			"segmentation_mask": class_mask,
			"pupil_mask": pupil_mask,
			"pupil_overlay_bgr": overlay_bgr,
			"pupil_class_id": class_id,
			"confidence": float(confidence),
			"used_fallback": used_fallback,
			"metrics": metrics,
		}

	def assess_pupil_dilation(
		self,
		baseline_bgr: np.ndarray,
		probe_bgr: np.ndarray,
		change_threshold_ratio: float = 0.08,
	) -> Dict[str, Any]:
		"""
		Compare two eye images and estimate pupil dilation change.
		"""
		base = self.detect_pupil(baseline_bgr)
		probe = self.detect_pupil(probe_bgr)

		d1 = float(base["metrics"]["diameter_px"])
		d2 = float(probe["metrics"]["diameter_px"])
		if d1 <= 0.0 or d2 <= 0.0:
			return {
				"status": "unknown",
				"reason": "pupil_not_detected",
				"baseline": base,
				"probe": probe,
			}

		ratio_change = (d2 - d1) / max(d1, 1e-6)
		if ratio_change > change_threshold_ratio:
			status = "dilated"
		elif ratio_change < -change_threshold_ratio:
			status = "constricted"
		else:
			status = "stable"

		return {
			"status": status,
			"diameter_change_ratio": float(ratio_change),
			"threshold": float(change_threshold_ratio),
			"baseline": base,
			"probe": probe,
		}

	def detect_vein_flow_change(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
		"""
		Placeholder for future liveness feature:
		detect subtle vessel changes tied to blood flow dynamics.
		"""
		return {
			"implemented": False,
			"message": "Vein-flow liveness detection is not implemented yet.",
		}

	def process_for_liveness(self, image_bgr: np.ndarray) -> Dict[str, Any]:
		"""
		Explicit two-step liveness flow:
		1) segmentation model pass
		2) pupil liveness process
		"""
		return self.detect_pupil(image_bgr)

