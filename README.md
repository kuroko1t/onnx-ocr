# PGNet ONNX Inference (Based on PaddleOCR)

This is PGNET's onnxruntime inference implementation of [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR).

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python inference_pgnet.py pgnet.onnx img_path
```

## Result

| Original  | Result |
| ------------- | ------------- |
| https://github.com/kuroko1t/onnx-ocr/blob/media/media/tullys.jpg  |  https://github.com/kuroko1t/onnx-ocr/blob/media/media/tullys_pgnet.jpg |

