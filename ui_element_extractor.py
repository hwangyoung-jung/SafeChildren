#!/usr/bin/env python3
"""Extract UI element labels and bounding boxes from a desktop screenshot."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import pytesseract


@dataclass
class Box:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    def to_list(self) -> List[int]:
        return [int(self.xmin), int(self.ymin), int(self.xmax), int(self.ymax)]

    def center(self) -> Tuple[float, float]:
        return ((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)


def load_image(path: str):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return image


def detect_checkboxes(image) -> List[Box]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    checkboxes: List[Box] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 8 or h < 8:
            continue
        if w > 40 or h > 40:
            continue
        aspect = w / float(h)
        if 0.8 <= aspect <= 1.2:
            area = cv2.contourArea(contour)
            if area < 40:
                continue
            checkboxes.append(Box(x, y, x + w, y + h))
    return sorted(checkboxes, key=lambda b: (b.ymin, b.xmin))


def detect_text_boxes(image) -> List[Tuple[str, Box]]:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    data = pytesseract.image_to_data(rgb, output_type=pytesseract.Output.DICT)
    text_boxes: List[Tuple[str, Box]] = []
    for i, text in enumerate(data["text"]):
        if not text.strip():
            continue
        x, y, w, h = (
            data["left"][i],
            data["top"][i],
            data["width"][i],
            data["height"][i],
        )
        text_boxes.append((text.strip(), Box(x, y, x + w, y + h)))
    return text_boxes


def group_labels(checkboxes: List[Box], text_boxes: List[Tuple[str, Box]]) -> List[dict]:
    results: List[dict] = []
    for checkbox in checkboxes:
        cx, cy = checkbox.center()
        candidates: List[Tuple[str, Box]] = []
        for text, box in text_boxes:
            _, ty = box.center()
            if box.xmin < checkbox.xmax:
                continue
            if abs(ty - cy) > 15:
                continue
            candidates.append((text, box))
        if not candidates:
            results.append({"label": "", "bounding_box": checkbox.to_list()})
            continue
        candidates.sort(key=lambda item: item[1].xmin)
        label = " ".join(text for text, _ in candidates)
        label_box = candidates[0][1]
        for _, box in candidates[1:]:
            label_box = Box(
                xmin=min(label_box.xmin, box.xmin),
                ymin=min(label_box.ymin, box.ymin),
                xmax=max(label_box.xmax, box.xmax),
                ymax=max(label_box.ymax, box.ymax),
            )
        results.append({"label": label, "bounding_box": label_box.to_list()})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract UI labels and bounding boxes from an image.")
    parser.add_argument("image", help="Path to the input image.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    args = parser.parse_args()

    image = load_image(args.image)
    checkboxes = detect_checkboxes(image)
    text_boxes = detect_text_boxes(image)
    results = group_labels(checkboxes, text_boxes)

    output = {"elements": results}
    if args.pretty:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
