# Fake Image & Photo Tampering Detector

An AI-powered image forensics system that analyzes a single image (`.jpg`, `.png`, `.webp`, `.tif`) and returns:

1. **Verdict:** Authentic or Tampered, based on an image-level tampering score.
2. **Tampering mask / heatmap:** A pixel-level overlay highlighting potentially manipulated regions.
3. **ELA visualization:** An Error Level Analysis map showing JPEG compression inconsistencies.

The system is designed to detect image splicing and copy-move manipulations by analyzing pixel-level inconsistencies rather than identifying the semantic content of an image.

**Note:** Detection performance depends on image quality, compression history, manipulation type, and similarity to the training dataset. The system does not guarantee detection of every manipulated image.

## How it works

```text
image ──► ELA (re-save at JPEG q=90, |orig − resaved| × 15)  ──┐
   └──────────────────────── RGB ─────────────────────────────┴─► 6-channel stack
        ──► U-Net (ResNet34 ImageNet encoder, SMP) ──► per-pixel P(tampered)
        ──► mask (P > 0.5) + image score = mean of top 0.5% pixel probabilities
        ──► verdict = score ≥ threshold (threshold calibrated on validation split)
```

### Design choices

- **ELA runs at original resolution, before resizing.** Resizing can smear the 8×8 JPEG block patterns that ELA measures.
- **Fixed ELA scale instead of per-image max-stretching.** The network sees error levels on a consistent scale across images. The application stretches ELA only for visualization.
- **RGB + ELA together (6 channels).** ELA alone can misinterpret edges and flat regions. RGB provides additional image context. SMP initializes the extra three input filters using the pretrained RGB filters.
- **No colour, blur or JPEG augmentation.** These transformations could introduce inconsistencies between RGB and precomputed ELA. Only flips and rotations are used, applied jointly to the image, ELA and mask.
- **BCE + Dice loss.** BCE penalizes pixel-level prediction errors, while Dice loss helps address the imbalance between manipulated regions and the authentic background.
- **Robust peak score.** The mean of the top 0.5% of pixel probabilities is used instead of the single maximum probability, reducing sensitivity to isolated high-probability pixels.
- **Threshold calibrated on validation only.** The classification threshold is selected using validation data and kept fixed during test evaluation to avoid test-set leakage.

## Project layout

| File | Purpose |
|---|---|
| `tamper_detector/ela.py` | Error Level Analysis |
| `tamper_detector/preprocess.py` | RGB + ELA preprocessing into tensors, shared by training and inference |
| `tamper_detector/data.py` | CASIA v2.0 and generic-folder discovery, stratified splitting and dataset handling |
| `tamper_detector/model.py` | U-Net builder, BCE + Dice loss, image scoring and checkpoint management |
| `tamper_detector/metrics.py` | Pixel F1/IoU and image ROC-AUC/accuracy |
| `tamper_detector/inference.py` | `TamperDetector.predict(img)` and heatmap overlay generation |
| `train.py` / `evaluate.py` / `predict.py` | Training, evaluation and prediction CLI entry points |
| `app.py` | Gradio web application |
| `tools/make_synthetic_dataset.py` | Synthetic CASIA-style dataset generation for pipeline testing |
| `notebooks/colab_train_and_demo.ipynb` | End-to-end Google Colab notebook (T4 GPU) |

## Quick start (Google Colab — no local GPU or installation required)

You need a Google account (for Colab) and a free Kaggle account (to download the dataset). Total execution time is approximately 30–45 minutes, mostly depending on training time and available GPU resources.

1. **Unzip the project.** Extract `tamper-detector.zip` on your computer. Keep the original ZIP file because you will upload it in step 5.
2. **Get a Kaggle API token.** Sign in at [Kaggle](https://www.kaggle.com/), go to *Settings → API → Create New Token*, and save the downloaded `kaggle.json`.
3. **Open the notebook in Colab.** Visit [Google Colab](https://colab.research.google.com/), choose *File → Upload notebook*, and select `tamper-detector/notebooks/colab_train_and_demo.ipynb`.
4. **Enable GPU acceleration.** Choose *Runtime → Change runtime type → T4 GPU → Save*.
5. **Run the notebook cells from top to bottom** (Shift + Enter on each cell). Two cells will display upload buttons:
   - Cell 1: Upload `tamper-detector.zip`.
   - Cell 2: Upload `kaggle.json`.
6. **Train the model.** The training cell prints one line per epoch. Once training finishes, the evaluation cell displays the metrics and example predictions.
7. **Try the web demo.** The last cell generates a public `https://….gradio.live` link. Open it, upload an image, and view the verdict, ELA visualization and tampering heatmap. The link remains active while the notebook cell is running.

**Important:** If Colab disconnects, files stored in `/content` are lost. You will need to rerun the notebook from the beginning.

To preserve your trained model, download `runs/casia_r34/best.pt` using the Colab file browser on the left. To reuse a saved model later, skip the training cell and point the remaining notebook cells to the downloaded checkpoint.

## Running locally

Install the project dependencies:

```bash
pip install -r requirements.txt
```

### 1. Test the pipeline using synthetic forgeries

Generate a small synthetic dataset:

```bash
python tools/make_synthetic_dataset.py --out data/synthetic --n 150
```

Train a model on the generated dataset:

```bash
python train.py --data-root data/synthetic --epochs 6 --img-size 256 --batch-size 8 --out runs/synthetic
```

This provides a quick way to verify the data pipeline, training and evaluation without downloading CASIA.

### 2. Train on CASIA v2.0

Point `--data-root` to a directory containing `Au/`, `Tp/` and a ground-truth mask folder:

```bash
python train.py --dataset casia --data-root /path/to/CASIA2 --epochs 20 --img-size 256 --out runs/casia_r34
```

### 3. Evaluate on the held-out test split

This uses the split saved during training and generates `metrics.json` along with example prediction panels:

```bash
python evaluate.py --checkpoint runs/casia_r34/best.pt --split runs/casia_r34/split.json
```

### 4. Cross-dataset evaluation

Evaluate generalization on the Columbia Uncompressed Image Splicing dataset using image-level metrics:

```bash
python evaluate.py --checkpoint runs/casia_r34/best.pt --authentic-dir columbia/4cam_auth --tampered-dir columbia/4cam_splc
```

### 5. Predict on an individual image

Run inference on your own image:

```bash
python predict.py --checkpoint runs/casia_r34/best.pt my_edit.jpg --out predictions
```

### 6. Launch the Gradio web application

```bash
python app.py --checkpoint runs/casia_r34/best.pt
```

### Useful flags for experimentation and ablation studies

- `--input-mode rgb|ela|rgb+ela`: Compare the contribution of RGB and ELA inputs.
- `--encoder efficientnet-b0`: Experiment with a different encoder.
- `--arch unetplusplus|fpn|deeplabv3plus`: Experiment with different segmentation architectures.
- `--img-size 384`: Use a higher input resolution, with increased computational cost.
- `--ela-quality 75|90|95`: Experiment with different JPEG qualities for ELA generation.

`train.py` saves:

- `best.pt`: Best checkpoint selected using validation pixel F1.
- `split.json`: Training, validation and test split information.
- `history.json`: Per-epoch training loss, F1, IoU, ROC-AUC and accuracy for analysis and plotting.

## Evaluation metrics

| Metric | Level | Computed on |
|---|---|---|
| Pixel F1 | Pixel | Tampered images only; averaged across images |
| Pixel IoU (Jaccard) | Pixel | Tampered images only; averaged across images |
| Image ROC-AUC | Image | Every image in the evaluation split; threshold-independent |
| Image accuracy | Image | Every image using the threshold calibrated on validation data |

**Pixel-level metrics** evaluate localization quality by measuring how accurately the predicted mask identifies manipulated regions.

- **Pixel F1:** Measures the balance between precision and recall of the predicted tampered regions.
- **Pixel IoU:** Measures the overlap between predicted and ground-truth tampered regions.

**Image-level metrics** evaluate the model's ability to distinguish authentic images from tampered images.

- **Image ROC-AUC:** Measures how effectively the model ranks tampered images above authentic images across classification thresholds.
- **Image accuracy:** Measures classification correctness using the fixed validation-calibrated threshold.

Pixel-level metrics are averaged over tampered images only. Image-level metrics are calculated using all images in the evaluation split.

The classification threshold is selected using validation predictions and remains fixed during test evaluation.

The evaluation script also calculates `image_acc_best`, which represents the highest accuracy obtained by searching thresholds on the evaluated predictions. This is a diagnostic metric and must not be presented as the primary held-out test accuracy because the threshold is optimized on the same evaluation data.

## Datasets

### CASIA v2.0

- Contains approximately 7.5k authentic and 5.1k tampered images, with ground-truth masks.
- On Kaggle, search for **CASIA 2.0 Image Tampering Detection Dataset**.
- The Colab notebook uses the dataset slug `divg07/casia-20-image-tampering-detection-dataset`. Replace it if the dataset location changes.
- Tampered images without a matching `_gt` mask are skipped, and their number is logged.

### Columbia Uncompressed Image Splicing

Used for cross-dataset generalization evaluation.

- Its `edgemask` files are colour-coded rather than binary.
- Convert them into binary masks before passing them through `--mask-dir`.
- Without ground-truth masks, evaluation provides image-level ROC-AUC and accuracy only.

## Verified results

### CASIA v2.0 — Held-out test evaluation

The trained model was evaluated on the held-out CASIA v2.0 test split using the best checkpoint selected during training.

| Metric | Test result |
|---|---:|
| Pixel F1 | **0.5650** |
| Pixel IoU | **0.5094** |
| Image ROC-AUC | **0.9607** |
| Image accuracy | **0.8921** |

**Evaluation details:**

| Detail | Value |
|---|---:|
| Total test images | 1,261 |
| Tampered images | 512 |
| Authentic images | 749 |
| Image classification threshold | 0.008762 |

**Important evaluation notes:**

- Pixel F1 and IoU are calculated on tampered images only, using a pixel probability threshold of 0.5.
- Image ROC-AUC and accuracy use all 1,261 test images.
- Image accuracy uses the classification threshold calibrated on the validation split.
- The evaluation script also calculated a best-threshold image accuracy of 0.8953. This is a diagnostic value only and is not the primary test accuracy.

The results demonstrate strong image-level discrimination on the held-out CASIA test split, while pixel-level localization remains a more challenging task.

### Synthetic dataset — Pipeline verification

The complete pipeline, including data discovery, training, evaluation, CLI prediction and the Gradio application, was also tested using a synthetic dataset.

With six epochs of training on CPU/MPS, from scratch and without ImageNet pretrained weights, the synthetic test evaluation achieved:

| Metric | Synthetic test result |
|---|---:|
| Pixel F1 | 0.86 |
| Pixel IoU | 0.77 |
| Image ROC-AUC | 1.00 |

These results demonstrate that the implementation can learn patterns in the generated synthetic forgeries and that the complete pipeline functions as intended.

**These are synthetic results, not CASIA results.** Synthetic forgeries are generally easier to distinguish than real-world manipulations. They should not be interpreted as real-world detection performance or compared directly with the CASIA evaluation.

## Limitations

- **Re-compression can erase traces.** Screenshots, WhatsApp and social-media re-uploads, and repeated saving as JPEG or WebP can weaken ELA cues. During testing, a tampered image re-saved as lossy WebP was classified as authentic.

- **Diffusion-based inpainting.** Generative inpainting that regenerates and re-encodes an entire image may leave no detectable ELA inconsistency. CASIA v2.0 does not adequately represent such manipulations, so detecting them requires additional training data containing these examples.

- **Small manipulations in high-resolution images.** Large images are resized to `--img-size` before network inference. Consequently, very small edited regions may be missed. Tiled inference is a potential future improvement.

- **Dataset bias and generalization.** CASIA has known dataset biases. Authentic and tampered images may differ in JPEG compression settings, image dimensions and other characteristics. High in-dataset performance may partly reflect these differences. Cross-dataset evaluation is therefore important when assessing generalization.

- **Not a universal authenticity guarantee.** The system identifies patterns associated with the manipulations represented in its training data. A Tampered verdict is not definitive proof of editing, and an Authentic verdict does not guarantee that an image is genuine.