import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from skimage import img_as_float
from skimage.filters import frangi

from pipeline.segmentation import seg_model


class EyeLivenessDetector:
	"""
	Liveness detector for eye images.

	Features:
	1) Vessel-based liveness: multi-frame blood-flow temporal analysis (PRIMARY)
	2) Pupil dilation liveness: multi-frame pupil size comparison (FALLBACK)
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

	# ----------------------------------------------------------------
	# Vessel-based liveness  (blood-flow temporal analysis)
	# ----------------------------------------------------------------

	def _extract_vessel_contrast_ratio(
		self,
		image_rgb: np.ndarray,
		sclera_mask: np.ndarray,
	) -> Optional[float]:
		"""
		Compute mean(vessel px intensity) / mean(non-vessel sclera px intensity)
		for one frame.

		Why a ratio and not raw variance:
		  Camera shake moves ALL sclera pixels together, keeping this ratio
		  stable.  Only actual blood flow changes vessel pixel brightness
		  independently of the background, making the temporal std of this
		  ratio a reliable live-vs-spoof discriminator.
		"""
		if np.count_nonzero(sclera_mask) < 500:
			return None

		gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
		gray_norm = cv2.normalize(
			gray, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F
		)

		# Frangi vesselness -- run only inside sclera region to avoid
		# noise from surrounding skin / lash regions.
		sclera_float = np.where(sclera_mask > 0, gray_norm, 0.0).astype(np.float32)
		v = frangi(img_as_float(sclera_float), sigmas=range(1, 4))
		v = np.nan_to_num(v, nan=0.0)
		v_norm = cv2.normalize(
			v.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX
		)

		# Adaptive vessel threshold: pixels above mean + 0.35 * std
		sclera_px = v_norm[sclera_mask > 0]
		mean_v = float(np.mean(sclera_px))
		std_v = float(np.std(sclera_px))
		vessel_threshold = mean_v + 0.35 * std_v

		vessel_mask = (v_norm > vessel_threshold) & (sclera_mask > 0)
		background_mask = (sclera_mask > 0) & ~vessel_mask

		if np.count_nonzero(vessel_mask) < 50 or np.count_nonzero(background_mask) < 50:
			return None

		vessel_mean = float(np.mean(gray[vessel_mask]))
		bg_mean = float(np.mean(gray[background_mask]))

		if bg_mean < 1.0:
			return None

		return vessel_mean / bg_mean

	def _vein_fallback_pupil(self, frames: List[np.ndarray]) -> Dict[str, Any]:
		"""Fall back to pupil dilation when vessel analysis is inconclusive."""
		if len(frames) < 2:
			return {
				"implemented": True,
				"status": "unknown",
				"liveness": None,
				"message": "Vein-flow inconclusive; insufficient frames for pupil fallback.",
				"fallback_used": True,
				"fallback_reason": "single_frame",
			}
		try:
			dilation = self.assess_pupil_dilation(frames[0], frames[-1])
			pd_status = dilation.get("status", "unknown")
			liveness = pd_status in ("dilated", "constricted")
			ratio = dilation.get("diameter_change_ratio")
			change_str = (
				f" (change={ratio:.4f})" if isinstance(ratio, float) else ""
			)
			return {
				"implemented": True,
				"status": f"pupil_{pd_status}",
				"liveness": liveness,
				"message": (
					f"Vein-flow inconclusive; pupil fallback: {pd_status}{change_str}."
				),
				"fallback_used": True,
				"fallback_reason": None,
				"pupil_dilation": {
					"status": pd_status,
					"diameter_change_ratio": ratio,
				},
			}
		except Exception as exc:
			return {
				"implemented": True,
				"status": "error",
				"liveness": None,
				"message": f"Both vein-flow and pupil fallback failed: {exc}",
				"fallback_used": True,
				"fallback_reason": "exception",
			}

	def detect_vein_flow_change(self, frames: Any, **kwargs: Any) -> Dict[str, Any]:
		"""
		Multi-frame blood-vessel liveness detection.

		Algorithm:
		  Per frame:
		    1. Segment -> sclera mask  (class == 1 from seg model)
		    2. Frangi vesselness on sclera pixels -> vessel map
		    3. vessel_contrast_ratio = mean(vessel px) / mean(background sclera px)
		  Across frames:
		    4. temporal_variance = std(contrast_ratios)
		       - Live:  blood flow changes vessel brightness independently
		                -> temporal_variance > LIVE_THRESHOLD  (0.005)
		       - Spoof: static image, no biology
		                -> temporal_variance < SPOOF_THRESHOLD (0.001)
		       - Borderline -> automatic pupil dilation fallback
		"""
		# Accept list/tuple of BGR frames (API usage) or single ndarray
		if isinstance(frames, (list, tuple)):
			frame_list = [f for f in frames if isinstance(f, np.ndarray)]
		elif isinstance(frames, np.ndarray):
			frame_list = [frames]
		else:
			frame_list = []

		print(f"[vein-flow] called with {len(frame_list)} frame(s)")

		if len(frame_list) < 2:
			return {
				"implemented": True,
				"status": "insufficient_frames",
				"liveness": None,
				"message": "Need at least 2 frames for vein-flow liveness.",
				"frames_analyzed": len(frame_list),
				"fallback_used": False,
			}

		contrast_ratios: List[float] = []
		sclera_coverages: List[float] = []

		for idx, frame in enumerate(frame_list):
			try:
				seg = self._segment_first(frame)
				image_rgb = seg["image_rgb"]
				class_mask = seg["class_mask"]

				sclera_mask = (class_mask == 1).astype(np.uint8) * 255
				coverage = float(np.count_nonzero(sclera_mask)) / float(sclera_mask.size)
				sclera_coverages.append(coverage)

				ratio = self._extract_vessel_contrast_ratio(image_rgb, sclera_mask)
				if ratio is not None:
					contrast_ratios.append(ratio)
					print(
						f"[vein-flow] frame {idx}: cov={coverage:.3f}"
						f" ratio={ratio:.6f}"
					)
				else:
					print(
						f"[vein-flow] frame {idx}: cov={coverage:.3f}"
						" vessel=None"
					)
			except Exception as exc:
				print(f"[vein-flow] frame {idx} error: {exc}")

		if len(contrast_ratios) < 2:
			print(
				f"[vein-flow] {len(contrast_ratios)} usable frame(s)"
				" -- falling back to pupil dilation"
			)
			result = self._vein_fallback_pupil(frame_list)
			result["fallback_reason"] = "insufficient_vessel_coverage"
			return result

		ratios = np.array(contrast_ratios, dtype=np.float64)
		temporal_variance = float(np.std(ratios))
		mean_ratio = float(np.mean(ratios))
		max_frame_diff = float(np.max(np.abs(np.diff(ratios))))

		LIVE_THRESHOLD = 0.005
		SPOOF_THRESHOLD = 0.001

		print(
			f"[vein-flow] temporal_variance={temporal_variance:.7f}"
			f"  mean_ratio={mean_ratio:.6f}"
			f"  max_diff={max_frame_diff:.7f}"
			f"  usable={len(contrast_ratios)}/{len(frame_list)}"
		)

		if temporal_variance < SPOOF_THRESHOLD:
			status, liveness = "spoof_detected", False
			message = (
				f"Near-zero vessel contrast variation (std={temporal_variance:.7f});"
				" likely static image."
			)
		elif temporal_variance >= LIVE_THRESHOLD:
			status, liveness = "live", True
			message = (
				f"Vessel contrast temporal variation detected"
				f" (std={temporal_variance:.7f}); live tissue confirmed."
			)
		else:
			print(
				f"[vein-flow] borderline variance={temporal_variance:.7f}"
				" -- falling back to pupil dilation"
			)
			result = self._vein_fallback_pupil(frame_list)
			result["fallback_reason"] = "borderline_vessel_variance"
			result["vein_temporal_variance"] = temporal_variance
			result["vein_mean_contrast_ratio"] = mean_ratio
			return result

		print(f"[vein-flow] result: {status}  liveness={liveness}")

		return {
			"implemented": True,
			"status": status,
			"liveness": liveness,
			"message": message,
			"temporal_variance": temporal_variance,
			"mean_contrast_ratio": mean_ratio,
			"max_frame_diff": max_frame_diff,
			"frames_analyzed": len(contrast_ratios),
			"total_frames": len(frame_list),
			"mean_sclera_coverage": (
				float(np.mean(sclera_coverages)) if sclera_coverages else 0.0
			),
			"fallback_used": False,
		}

	def process_for_liveness(self, image_bgr: np.ndarray) -> Dict[str, Any]:
		"""Single-image path: segment then detect pupil."""
		return self.detect_pupil(image_bgr)

