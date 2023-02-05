import argparse
import os

import cv2
import numpy as np
import onnxruntime
from paddleocr.ppocr.data.imaug.operators import (E2EResizeForTest, KeepKeys,
                                                  NormalizeImage, ToCHWImage)
from paddleocr.ppocr.postprocess.pg_postprocess import PGPostProcess

from pgnet.chr_dct import chr_dct_list


class PGNetPredictor:
    def __init__(self, img_path, cpu):
        self.img_path = img_path
        self.dict_path = "ic15_dict.txt"
        if not os.path.exists(self.dict_path):
            with open(self.dict_path, "w") as f:
                f.writelines(chr_dct_list)
        if not cpu:
            providers = ["CUDAExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        self.sess = onnxruntime.InferenceSession(args.model_path, providers=providers)

    def preprocess(self, img_path):
        img = cv2.imread(img_path)
        self.ori_im = img.copy()
        data = {"image": img}
        transforms = [
            E2EResizeForTest(max_side_len=768, valid_set="totaltext"),
            NormalizeImage(
                scale=1.0 / 255.0,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                order="hwc",
            ),
            ToCHWImage(),
            KeepKeys(keep_keys=["image", "shape"]),
        ]
        for transform in transforms:
            data = transform(data)
        img, shape_list = data
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        return img, shape_list

    def predict(self, img):
        ort_inputs = {self.sess.get_inputs()[0].name: img}
        outputs = self.sess.run(None, ort_inputs)
        preds = {}
        preds["f_border"] = outputs[0]
        preds["f_char"] = outputs[1]
        preds["f_direction"] = outputs[2]
        preds["f_score"] = outputs[3]
        return preds

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def postprocess(self, preds, shape_list):
        pgpostprocess = PGPostProcess(
            character_dict_path=self.dict_path,
            valid_set="totaltext",
            score_thresh=0.5,
            mode="fast",
        )
        post_result = pgpostprocess(preds, shape_list)
        points, strs = post_result["points"], post_result["texts"]
        dt_boxes = self.filter_tag_det_res_only_clip(points, self.ori_im.shape)
        return dt_boxes, strs

    def __call__(self):
        img, shape_list = self.preprocess(self.img_path)
        preds = self.predict(img)
        dt_boxes, strs = self.postprocess(preds, shape_list)
        return dt_boxes, strs

    def draw(self, dt_boxes, strs, img_path):
        src_im = cv2.imread(img_path)
        width, height, _ = src_im.shape
        for box, str in zip(dt_boxes, strs):
            box = box.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
            cv2.putText(
                src_im,
                str,
                org=(int(box[0, 0, 0]), int(box[0, 0, 1])),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.7 / 500 * width / 2,
                color=(0, 255, 0),
                thickness=int(1 / 1000 * width),
            )
        img_out_name = os.path.basename(img_path).split(".")[0]
        img_out_name = f"{img_out_name}_pgnet.jpg"
        cv2.imwrite(img_out_name, src_im)
        return src_im


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PGPNET inference")
    parser.add_argument("model_path", type=str, help="onnxmodel path")
    parser.add_argument("img_path", type=str, help="image path")
    parser.add_argument(
        "--cpu", action="store_true", help="cpu inference, default device is gpu"
    )
    args = parser.parse_args()
    pgnetpredictor = PGNetPredictor(args.img_path, args.cpu)
    dt_boxes, strs = pgnetpredictor()
    print(f"Predict string:{strs}")
    pgnetpredictor.draw(dt_boxes, strs, args.img_path)
