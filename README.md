# PGNet ONNX Inference (Based on PaddleOCR)

This is PGNET's onnxruntime inference implementation of [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR).

## Setup

```bash
pip install -r requirements.txt
wget https://github.com/kuroko1t/onnx-ocr/releases/download/0.1/pgnet.onnx
```

## Run

```bash
python inference_pgnet.py pgnet.onnx img_path
```

## Result

| Original  | Result |
| ------------- | ------------- |
| ![image0](https://github.com/kuroko1t/onnx-ocr/blob/media/media/tullys.jpg?raw=true)  | ![image1](https://github.com/kuroko1t/onnx-ocr/blob/media/media/tullys_pgnet.jpg?raw=true) |

