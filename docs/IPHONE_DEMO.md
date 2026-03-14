# Running the Drone Detection Model on iPhone

You have three main options, from most “native” to simplest to build.

---

## Option 1: **Core ML (on-device, recommended for demo)**

Run the model **on the iPhone** with Apple’s Core ML. Best balance of performance, battery, and offline use.

### Steps

1. **Export to Core ML**
   - You already have ONNX: `runs/binary_no_aug/best.onnx`.
   - Install: `pip install coremltools`
   - Convert ONNX → Core ML (Python):
     ```python
     import coremltools as ct
     model = ct.converters.onnx.convert(
         model_path="runs/binary_no_aug/best.onnx",
         minimum_deployment_target=ct.target.iOS16,
     )
     model.save("DroneClassifier.mlpackage")  # or .mlmodel
     ```
   - If the converter complains about ops, try exporting ONNX with a fixed batch size (no `--dynamic`) so input shape is `[1, 48000]`.

2. **Preprocessing on iOS**
   - The model expects **16 kHz, mono, 3 s (48,000 samples), float32, peak-normalized**.
   - In Swift: use **AVAudioEngine** or **AVAudioRecorder** to capture from the microphone.
   - Resample to 16 kHz if needed (AVAudioFormat + conversion), take 48,000 samples, convert to float array, peak-normalize (divide by max abs value), then pass to the Core ML model’s input.

3. **Swift / Xcode**
   - Add `DroneClassifier.mlpackage` to the app target.
   - Use the generated Swift class (e.g. `DroneClassifier`) and call `prediction(input: ...)` with a 48,000-element array (or whatever input name the model has).
   - Run this in a loop (e.g. every 0.1 s on a 3 s sliding window) and show “Drone” / “Noise” and optionally a confidence bar.

**Pros:** Native, fast, works offline, good battery.  
**Cons:** Need to implement or reuse 16 kHz capture + 3 s buffer + normalization in Swift.

---

## Option 2: **ONNX Runtime on iPhone (on-device, no conversion)**

Keep using the **same ONNX file** and run it with **ONNX Runtime** on iOS.

### Steps

1. **Add ONNX Runtime to the app**
   - Use **CocoaPods**: `pod 'onnxruntime-objective-c'` or the Swift Package at [onnxruntime-ios](https://github.com/microsoft/onnxruntime/tree/main/samples/ios).
   - Or build the iOS framework from the [ONNX Runtime repo](https://github.com/microsoft/onnxruntime) and embed it.

2. **Bundle the model**
   - Add `best.onnx` to the Xcode project so it’s in the app bundle.
   - Load it at runtime with the ONNX Runtime API.

3. **Audio + inference**
   - Same as Core ML: capture audio (e.g. AVAudioEngine), resample to 16 kHz, fill a 3 s buffer (48,000 samples), peak-normalize, copy to a float array, run `session.run()` with input name `"waveform"` and shape `[1, 48000]`.
   - Read the `"logits"` output, apply softmax, and use the “drone” class probability for your UI.

**Pros:** No conversion step; same ONNX you use elsewhere.  
**Cons:** Larger app size than Core ML; you must implement preprocessing in Swift/ObjC.

---

## Option 3: **Server-based demo (easiest to get running)**

The **iPhone app only records and sends audio**; a **server** runs the model and returns the result. Good for a quick demo or when you don’t need offline.

### Steps

1. **Backend**
   - Small API (e.g. **Flask** or **FastAPI**) that:
     - Accepts a 3 s (or 1 min) WAV (or raw float32) upload.
     - Loads your ONNX model (e.g. with `onnxruntime`).
     - Resamples to 16 kHz if needed, splits into 3 s windows, normalizes, runs inference.
     - Returns JSON, e.g. `{"prediction": "drone", "probability": 0.87}` or per-window results for a 1 min file.
   - Deploy on a VPS, **Railway**, **Render**, or **Google Cloud Run**.

2. **iPhone app**
   - Record 3 s (or stream chunks) using **AVAudioEngine**.
   - Encode as WAV or raw and **POST** to your API.
   - Display the returned label and confidence.

**Pros:** Easiest to implement; no ONNX/Core ML on the device; you can change the model without an app update.  
**Cons:** Requires network and a server; latency; not suitable for “always-on” offline use.

---

## Comparison

| Criteria           | Core ML           | ONNX Runtime iOS  | Server-based      |
|-------------------|-------------------|-------------------|-------------------|
| On-device         | Yes               | Yes               | No                |
| Offline           | Yes               | Yes               | No                |
| Conversion step  | ONNX → Core ML    | None              | None              |
| App size          | Small             | Larger            | Small             |
| Implementation   | Swift + AVFoundation + Core ML | Swift + AVFoundation + ORT | Swift + AVFoundation + HTTP |
| Ease of first demo| Medium            | Medium            | Easiest           |

---

## Minimal “what the model needs” (all options)

- **Input:** 16 kHz, mono, **48,000 samples** (3 s), **float32**, **peak-normalized** (divide by max of abs).
- **Output:** Logits shape `[1, 2]` (noise, drone); apply softmax for probabilities; class 1 = drone.

For a **sliding window** (e.g. every 0.1 s), keep a 48,000-sample ring buffer, run inference each hop, and show the current “drone” probability or a binary “Drone detected” when above a threshold.

---

## Suggested path for a demo

- **Quick demo / hackathon:** Option 3 (server + simple iOS app that records and POSTs).
- **Real on-device demo / no backend:** Option 1 (Core ML) if the ONNX → Core ML conversion succeeds; else Option 2 (ONNX Runtime on iOS).

If you tell me which option you prefer (Core ML, ONNX Runtime, or server), I can outline concrete code (e.g. a minimal Swift snippet for capture + inference or a minimal Flask/FastAPI endpoint).
