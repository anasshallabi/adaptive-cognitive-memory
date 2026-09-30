# ACM v0.2 — Real-image one-shot experiment

**Status:** code implemented; real-image benchmark results **not yet available**. This system does not perform human-like concept learning.

## Windows 11 / NVIDIA setup (also supports CPU)

Install Python 3.10+ and, in PowerShell:

```powershell
git clone https://github.com/anasshallabi/adaptive-cognitive-memory.git
cd adaptive-cognitive-memory
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install **PyTorch + torchvision** with the official CUDA/CPU selector: https://pytorch.org/get-started/locally/ . Then:

```powershell
python -m pip install open_clip_torch Pillow
python -c "import torch; print('CUDA:',torch.cuda.is_available())"
python -m unittest discover -s tests -v
```

Core tests require no ML packages. `--device cpu` is available without CUDA. Initial OpenCLIP initialization can download sizable pretrained weights and requires internet and free disk space; ACM does not upload local pictures to hosted inference APIs.

## One-shot demonstration

Use local photographs you have rights to use: a first Toyota car with readable logo (`data/toyota_a.jpg`), another physical Toyota car (`data/toyota_b.jpg`), and a different brand (`data/honda.jpg`).

```powershell
python -m examples.vision_one_shot --support data/toyota_a.jpg --label Toyota --query data/toyota_b.jpg data/honda.jpg --device auto --threshold 0.85
```

This prints JSON with model metadata and per-image cosine similarity. **The default threshold 0.85 is only an example**; tune thresholds on separate validation brands. Similarity is not a calibrated probability. Distinguish recognition based on visible logo/text from recognition based on body shape. The memory is in process only: information is lost on restart.

## Multi-brand evaluation

Example CSV `datasets/test_episode.csv`, with paths relative to the manifest:

```csv
path,label,split,vehicle_id,capture_group
images/a_toyota.jpg,Toyota,support,toyota_car_1,session_1
images/b_toyota.jpg,Toyota,known,toyota_car_2,session_2
images/c_honda.jpg,Honda,unknown,honda_car_1,session_3
```

```powershell
python -m examples.evaluate_images --manifest datasets/test_episode.csv --threshold 0.85 --device auto
```

Exactly one `support` per known label; `known` examples are distinct cars of learned brands, `unknown` labels are disjoint. No image path, vehicle ID or capture session may leak between splits. This enforcement relies on truthful annotations and cannot detect all duplicate/reused content.

Report known-class top-1 accuracy, unknown rejection rate, per-image predictions, CPU/GPU and model version; extend toward uncertainty, memory and latency profiling, repeated episodes, and cross-brand masking experiments. The evaluator is an **initial baseline**, not a full benchmark. Never calibrate with final test brands.

## Research caveats

The pretrained OpenCLIP encoder has already seen many images during pretraining; ACM only binds new labels to existing features. Equal model/backbone, data, thresholds and preprocessing are mandatory for comparisons. Do not commit private images, tokens or unlicensed third-party datasets.

Sources: [OpenCLIP upstream](https://github.com/mlfoundations/open_clip), [PyTorch official install](https://pytorch.org/get-started/locally/).
