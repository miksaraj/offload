//! YOLOv8 player detection via ONNX Runtime.
//!
//! Phase 2a: model loading, preprocessing, and raw inference.
//! Phase 2b: postprocessing — decoding the `[1, 84, N]` output tensor,
//! person-class and confidence filtering, NMS, and mapping boxes from the
//! 640×640 letterbox back to original frame pixels — plus
//! [`draw_detections`] for annotated debug frames.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use image::{imageops, ImageBuffer, Rgb, RgbImage};
use ndarray::{Array4, ArrayD, ArrayViewD, Axis};
use thiserror::Error;
use video_io::Frame;

#[derive(Debug, Error)]
pub enum DetectorError {
    #[error("model not found: {0} (run models/download.sh to fetch it)")]
    ModelNotFound(String),
    #[error("inference error: {0}")]
    Inference(String),
}

pub type Result<T> = std::result::Result<T, DetectorError>;

impl From<ort::Error> for DetectorError {
    fn from(err: ort::Error) -> Self {
        DetectorError::Inference(err.to_string())
    }
}

/// Axis-aligned box; `x`/`y` are the top-left corner.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Rect {
    pub fn area(&self) -> f32 {
        self.width.max(0.0) * self.height.max(0.0)
    }

    /// Intersection-over-union with `other`; 0 when either box is empty.
    pub fn iou(&self, other: &Rect) -> f32 {
        let ix = (self.x + self.width).min(other.x + other.width) - self.x.max(other.x);
        let iy = (self.y + self.height).min(other.y + other.height) - self.y.max(other.y);
        let inter = ix.max(0.0) * iy.max(0.0);
        let union = self.area() + other.area() - inter;
        if union <= 0.0 {
            0.0
        } else {
            inter / union
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    /// Pixel coordinates in the original frame space.
    pub bbox: Rect,
    pub confidence: f32,
    /// COCO class id; 0 = person.
    pub class_id: u32,
}

#[derive(Debug, Clone)]
pub struct DetectorConfig {
    pub confidence_threshold: f32,
    pub nms_iou_threshold: f32,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.4,
            nms_iou_threshold: 0.45,
        }
    }
}

/// YOLOv8's expected square input resolution.
pub const MODEL_INPUT_SIZE: u32 = 640;
/// COCO class id for "person" — the only class Offload keeps.
pub const PERSON_CLASS_ID: u32 = 0;
/// Upper bound on detections kept per frame after NMS (ultralytics' `max_det`).
pub const MAX_DETECTIONS: usize = 300;
const LETTERBOX_FILL: u8 = 114;
const INPUT_TENSOR_NAME: &str = "images";
const OUTPUT_TENSOR_NAME: &str = "output0";

/// How a source frame was mapped into the 640×640 letterbox, so boxes can be
/// mapped back: `letterbox = source * scale + pad`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LetterboxTransform {
    pub scale: f32,
    pub pad_x: f32,
    pub pad_y: f32,
    pub source_width: u32,
    pub source_height: u32,
}

impl LetterboxTransform {
    pub fn for_frame(width: u32, height: u32) -> Self {
        let scale =
            (MODEL_INPUT_SIZE as f32 / width as f32).min(MODEL_INPUT_SIZE as f32 / height as f32);
        let resized_w = ((width as f32 * scale).round() as u32).max(1);
        let resized_h = ((height as f32 * scale).round() as u32).max(1);
        Self {
            scale,
            pad_x: ((MODEL_INPUT_SIZE - resized_w) / 2) as f32,
            pad_y: ((MODEL_INPUT_SIZE - resized_h) / 2) as f32,
            source_width: width,
            source_height: height,
        }
    }

    /// Maps a box in letterbox pixels to source-frame pixels, clamped to
    /// the frame.
    pub fn to_source(&self, r: &Rect) -> Rect {
        let w = self.source_width as f32;
        let h = self.source_height as f32;
        let x0 = ((r.x - self.pad_x) / self.scale).clamp(0.0, w);
        let y0 = ((r.y - self.pad_y) / self.scale).clamp(0.0, h);
        let x1 = ((r.x + r.width - self.pad_x) / self.scale).clamp(0.0, w);
        let y1 = ((r.y + r.height - self.pad_y) / self.scale).clamp(0.0, h);
        Rect {
            x: x0,
            y: y0,
            width: x1 - x0,
            height: y1 - y0,
        }
    }
}

/// Resizes a frame to [`MODEL_INPUT_SIZE`] square, preserving aspect ratio and
/// padding the remainder with grey (114,114,114) — the standard YOLO letterbox.
fn letterbox(frame: &Frame) -> RgbImage {
    let img: ImageBuffer<Rgb<u8>, &[u8]> =
        ImageBuffer::from_raw(frame.width, frame.height, frame.pixels.as_slice())
            .expect("Frame pixel buffer length must be width * height * 3");

    let t = LetterboxTransform::for_frame(frame.width, frame.height);
    let resized_w = ((frame.width as f32 * t.scale).round() as u32).max(1);
    let resized_h = ((frame.height as f32 * t.scale).round() as u32).max(1);
    let resized = imageops::resize(&img, resized_w, resized_h, imageops::FilterType::Triangle);

    let mut canvas: RgbImage =
        ImageBuffer::from_pixel(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, Rgb([LETTERBOX_FILL; 3]));
    imageops::replace(&mut canvas, &resized, t.pad_x as i64, t.pad_y as i64);
    canvas
}

/// Normalises `[0, 255] -> [0.0, 1.0]` and converts `HWC -> NCHW`.
fn hwc_to_nchw(img: &RgbImage) -> Array4<f32> {
    Array4::from_shape_fn(
        (1, 3, MODEL_INPUT_SIZE as usize, MODEL_INPUT_SIZE as usize),
        |(_, c, y, x)| img.get_pixel(x as u32, y as u32)[c] as f32 / 255.0,
    )
}

/// Preprocesses a frame into the `[1, 3, 640, 640]` float32 tensor YOLOv8 expects.
pub fn preprocess(frame: &Frame) -> Array4<f32> {
    hwc_to_nchw(&letterbox(frame))
}

/// Decodes a raw YOLOv8 output tensor (`[1, 4 + classes, anchors]`; rows are
/// `cx, cy, w, h` in letterbox pixels followed by per-class scores, with no
/// separate objectness score) into person [`Detection`]s in source-frame
/// pixels.
///
/// Mirrors ultralytics' single-label postprocessing: each anchor takes its
/// highest-scoring class; anchors whose best class isn't person, or whose
/// score is below `confidence_threshold`, are dropped; greedy NMS at
/// `nms_iou_threshold` runs in letterbox space; survivors (at most
/// [`MAX_DETECTIONS`], highest confidence first) are then mapped back
/// through `transform`.
pub fn decode_output(
    output: ArrayViewD<'_, f32>,
    transform: &LetterboxTransform,
    config: &DetectorConfig,
) -> Result<Vec<Detection>> {
    let shape = output.shape();
    if shape.len() != 3 || shape[0] != 1 || shape[1] <= 4 {
        return Err(DetectorError::Inference(format!(
            "unexpected detector output shape {shape:?}; expected [1, 4 + classes, anchors]"
        )));
    }
    let preds = output.index_axis(Axis(0), 0); // [4 + classes, anchors]

    let mut candidates = Vec::new();
    for anchor in preds.axis_iter(Axis(1)) {
        let (best_class, best_score) =
            anchor
                .iter()
                .skip(4)
                .enumerate()
                .fold(
                    (0usize, f32::MIN),
                    |best, (i, &s)| {
                        if s > best.1 {
                            (i, s)
                        } else {
                            best
                        }
                    },
                );
        if best_class as u32 != PERSON_CLASS_ID || best_score < config.confidence_threshold {
            continue;
        }
        let (cx, cy, w, h) = (anchor[0], anchor[1], anchor[2], anchor[3]);
        candidates.push(Detection {
            bbox: Rect {
                x: cx - w / 2.0,
                y: cy - h / 2.0,
                width: w,
                height: h,
            },
            confidence: best_score,
            class_id: PERSON_CLASS_ID,
        });
    }

    let mut kept = nms(candidates, config.nms_iou_threshold);
    kept.truncate(MAX_DETECTIONS);
    for d in &mut kept {
        d.bbox = transform.to_source(&d.bbox);
    }
    kept.retain(|d| d.bbox.width > 0.0 && d.bbox.height > 0.0);
    Ok(kept)
}

/// Greedy non-maximum suppression: highest confidence first, dropping any
/// box whose IoU with an already-kept box exceeds `iou_threshold`.
pub fn nms(mut detections: Vec<Detection>, iou_threshold: f32) -> Vec<Detection> {
    detections.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
    let mut kept: Vec<Detection> = Vec::with_capacity(detections.len().min(MAX_DETECTIONS));
    for d in detections {
        if kept.iter().all(|k| k.bbox.iou(&d.bbox) <= iou_threshold) {
            kept.push(d);
        }
    }
    kept
}

/// Copies `frame` and draws each detection's box on it (debug output,
/// SPEC.md §5 `debug/detection/`). Box colour runs from red (low
/// confidence) to green (high); line width scales with frame size.
pub fn draw_detections(frame: &Frame, detections: &[Detection]) -> RgbImage {
    let mut img: RgbImage = ImageBuffer::from_raw(frame.width, frame.height, frame.pixels.clone())
        .expect("Frame pixel buffer length must be width * height * 3");
    let thickness = (frame.width.max(frame.height) / 400).max(2);
    for d in detections {
        let c = d.confidence.clamp(0.0, 1.0);
        let colour = Rgb([((1.0 - c) * 255.0) as u8, (c * 255.0) as u8, 0]);
        draw_rect(&mut img, &d.bbox, thickness, colour);
    }
    img
}

fn draw_rect(img: &mut RgbImage, r: &Rect, thickness: u32, colour: Rgb<u8>) {
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 {
        return;
    }
    let x0 = (r.x.max(0.0) as u32).min(w - 1);
    let y0 = (r.y.max(0.0) as u32).min(h - 1);
    let x1 = ((r.x + r.width).max(0.0) as u32).min(w - 1);
    let y1 = ((r.y + r.height).max(0.0) as u32).min(h - 1);
    for t in 0..thickness {
        for x in x0..=x1 {
            for y in [y0.saturating_add(t).min(y1), y1.saturating_sub(t).max(y0)] {
                img.put_pixel(x, y, colour);
            }
        }
        for y in y0..=y1 {
            for x in [x0.saturating_add(t).min(x1), x1.saturating_sub(t).max(x0)] {
                img.put_pixel(x, y, colour);
            }
        }
    }
}

/// Dynamically loads the ONNX Runtime shared library and commits a process-wide
/// `ort` environment, exactly once. The path comes from `ORT_DYLIB_PATH` since
/// `ort`'s `load-dynamic` feature (used because the default `download-binaries`
/// feature fetches a prebuilt library from a CDN this project can't assume is
/// reachable) has no built-in env var fallback — see CLAUDE.md's Gotchas.
fn ensure_ort_initialized() -> Result<()> {
    static INIT: OnceLock<std::result::Result<(), String>> = OnceLock::new();
    INIT.get_or_init(|| {
        let dylib_path = std::env::var("ORT_DYLIB_PATH").map_err(|_| {
            "ORT_DYLIB_PATH not set; point it at libonnxruntime.so (see .claude/skills/run-offload/SKILL.md)".to_string()
        })?;
        ort::init_from(dylib_path)
            .map_err(|e| e.to_string())?
            .commit();
        Ok(())
    })
    .clone()
    .map_err(DetectorError::Inference)
}

/// Loads a YOLOv8 ONNX model and runs person detection on frames.
pub struct Detector {
    session: ort::session::Session,
    config: DetectorConfig,
}

impl Detector {
    /// Loads the model at `model_path` and runs a warm-up inference so the
    /// first real `detect`/`run_inference` call isn't paying for lazy
    /// initialisation inside the timed pipeline.
    pub fn new(model_path: impl AsRef<Path>, config: DetectorConfig) -> Result<Self> {
        let model_path: PathBuf = model_path.as_ref().to_path_buf();
        if !model_path.exists() {
            return Err(DetectorError::ModelNotFound(
                model_path.display().to_string(),
            ));
        }

        ensure_ort_initialized()?;

        let mut session = ort::session::Session::builder()?.commit_from_file(&model_path)?;

        let warmup_input =
            Array4::<f32>::zeros((1, 3, MODEL_INPUT_SIZE as usize, MODEL_INPUT_SIZE as usize));
        let warmup_outputs = session.run(ort::inputs![
            INPUT_TENSOR_NAME => ort::value::TensorRef::from_array_view(&warmup_input)?
        ])?;
        let warmup_shape = warmup_outputs[OUTPUT_TENSOR_NAME]
            .try_extract_array::<f32>()?
            .shape()
            .to_vec();
        tracing::info!(
            model = %model_path.display(),
            warmup_output_shape = ?warmup_shape,
            "detector model loaded and warmed up"
        );
        drop(warmup_outputs);

        Ok(Self { session, config })
    }

    /// Preprocesses `frame`, runs raw inference, and returns the model's
    /// undecoded output tensor. Logs the tensor's shape.
    pub fn run_inference(&mut self, frame: &Frame) -> Result<ArrayD<f32>> {
        let input = preprocess(frame);
        let outputs = self.session.run(ort::inputs![
            INPUT_TENSOR_NAME => ort::value::TensorRef::from_array_view(&input)?
        ])?;
        let output = outputs[OUTPUT_TENSOR_NAME].try_extract_array::<f32>()?;
        tracing::trace!(shape = ?output.shape(), "raw detector output tensor");
        Ok(output.to_owned())
    }

    /// Detects people in `frame`: preprocess, infer, decode, NMS. Boxes are
    /// in `frame`'s pixel coordinates. An empty `Vec` is a normal result.
    pub fn detect(&mut self, frame: &Frame) -> Result<Vec<Detection>> {
        let raw = self.run_inference(frame)?;
        let transform = LetterboxTransform::for_frame(frame.width, frame.height);
        decode_output(raw.view(), &transform, &self.config)
    }

    pub fn config(&self) -> &DetectorConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    fn det(x: f32, y: f32, w: f32, h: f32, conf: f32) -> Detection {
        Detection {
            bbox: Rect {
                x,
                y,
                width: w,
                height: h,
            },
            confidence: conf,
            class_id: PERSON_CLASS_ID,
        }
    }

    /// Builds a `[1, 84, n]` tensor from `(cx, cy, w, h, class, score)` rows.
    fn tensor(anchors: &[(f32, f32, f32, f32, usize, f32)]) -> ArrayD<f32> {
        let mut t = Array3::<f32>::zeros((1, 84, anchors.len()));
        for (i, &(cx, cy, w, h, class, score)) in anchors.iter().enumerate() {
            t[[0, 0, i]] = cx;
            t[[0, 1, i]] = cy;
            t[[0, 2, i]] = w;
            t[[0, 3, i]] = h;
            t[[0, 4 + class, i]] = score;
        }
        t.into_dyn()
    }

    #[test]
    fn iou_basics() {
        let a = Rect {
            x: 0.0,
            y: 0.0,
            width: 10.0,
            height: 10.0,
        };
        let b = Rect {
            x: 5.0,
            y: 0.0,
            width: 10.0,
            height: 10.0,
        };
        let far = Rect {
            x: 100.0,
            y: 100.0,
            width: 1.0,
            height: 1.0,
        };
        assert!((a.iou(&a) - 1.0).abs() < 1e-6);
        assert!((a.iou(&b) - 50.0 / 150.0).abs() < 1e-6);
        assert_eq!(a.iou(&far), 0.0);
    }

    #[test]
    fn nms_keeps_highest_and_drops_overlaps_only() {
        let kept = nms(
            vec![
                det(0.0, 0.0, 10.0, 10.0, 0.6),
                det(1.0, 0.0, 10.0, 10.0, 0.9), // overlaps the first heavily
                det(50.0, 50.0, 10.0, 10.0, 0.5),
            ],
            0.45,
        );
        let confs: Vec<f32> = kept.iter().map(|d| d.confidence).collect();
        assert_eq!(confs, vec![0.9, 0.5]);
    }

    #[test]
    fn letterbox_round_trips_for_wide_frames() {
        // 1280x720 -> scale 0.5, resized 640x360, pad_y = 140.
        let t = LetterboxTransform::for_frame(1280, 720);
        assert!((t.scale - 0.5).abs() < 1e-6);
        assert_eq!((t.pad_x, t.pad_y), (0.0, 140.0));
        let lb = Rect {
            x: 100.0,
            y: 190.0,
            width: 50.0,
            height: 100.0,
        };
        let src = t.to_source(&lb);
        assert_eq!(
            src,
            Rect {
                x: 200.0,
                y: 100.0,
                width: 100.0,
                height: 200.0
            }
        );
    }

    #[test]
    fn letterbox_clamps_boxes_that_spill_into_padding() {
        let t = LetterboxTransform::for_frame(1280, 720);
        let src = t.to_source(&Rect {
            x: -20.0,
            y: 100.0,
            width: 60.0,
            height: 100.0,
        });
        assert_eq!(src.x, 0.0);
        assert_eq!(src.y, 0.0);
        assert!((src.width - 80.0).abs() < 1e-4);
        assert!((src.height - 120.0).abs() < 1e-4);
    }

    #[test]
    fn decode_filters_class_threshold_and_overlaps_then_maps_to_source() {
        let raw = tensor(&[
            (320.0, 320.0, 40.0, 80.0, 0, 0.90), // person, kept
            (322.0, 321.0, 40.0, 80.0, 0, 0.80), // duplicate of the above, NMS'd
            (100.0, 320.0, 40.0, 80.0, 0, 0.30), // person below threshold
            (500.0, 320.0, 40.0, 80.0, 5, 0.95), // bus, not a person
            (500.0, 400.0, 20.0, 40.0, 0, 0.50), // second person, kept
        ]);
        let t = LetterboxTransform::for_frame(1280, 720);
        let dets = decode_output(raw.view(), &t, &DetectorConfig::default()).unwrap();
        assert_eq!(dets.len(), 2);
        assert_eq!(dets[0].confidence, 0.90);
        assert_eq!(dets[1].confidence, 0.50);
        // (320 - 20, 320 - 40) in letterbox -> ((300 - 0) / 0.5, (280 - 140) / 0.5).
        assert_eq!(
            dets[0].bbox,
            Rect {
                x: 600.0,
                y: 280.0,
                width: 80.0,
                height: 160.0
            }
        );
        assert!(dets.iter().all(|d| d.class_id == PERSON_CLASS_ID));
    }

    #[test]
    fn decode_skips_anchors_whose_best_class_is_not_person() {
        // Person score clears the threshold, but another class scores higher.
        let mut raw = tensor(&[(320.0, 320.0, 40.0, 80.0, 0, 0.6)]);
        raw[[0, 4 + 2, 0]] = 0.7;
        let t = LetterboxTransform::for_frame(640, 640);
        assert!(decode_output(raw.view(), &t, &DetectorConfig::default())
            .unwrap()
            .is_empty());
    }

    #[test]
    fn decode_rejects_unexpected_shapes() {
        let raw = ArrayD::<f32>::zeros(vec![1, 4, 10]);
        let t = LetterboxTransform::for_frame(640, 640);
        assert!(decode_output(raw.view(), &t, &DetectorConfig::default()).is_err());
    }

    #[test]
    fn draw_detections_paints_box_edges_only() {
        let frame = Frame {
            timestamp_ms: 0,
            frame_number: 0,
            pixels: vec![0; 100 * 100 * 3],
            width: 100,
            height: 100,
        };
        let img = draw_detections(&frame, &[det(10.0, 10.0, 50.0, 50.0, 1.0)]);
        assert_eq!(*img.get_pixel(10, 30), Rgb([0, 255, 0]));
        assert_eq!(*img.get_pixel(60, 30), Rgb([0, 255, 0]));
        assert_eq!(*img.get_pixel(35, 35), Rgb([0, 0, 0]));
    }
}
