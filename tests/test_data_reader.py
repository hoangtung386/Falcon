from __future__ import annotations

import os
import tempfile

import pandas as pd
from falcon.data.io import AnnotType, PictureInfo, read_csv_annotation_file


class TestReadCSV:
    def test_read_csv_original(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "ann.csv")
            img_dir = os.path.join(tmpdir, "images")
            os.makedirs(img_dir)

            df = pd.DataFrame(
                {
                    "img_name": ["test.jpg"],
                    "face_x0": [0],
                    "face_y0": [0],
                    "face_x1": [100],
                    "face_y1": [100],
                    "age": [25],
                    "gender": ["M"],
                }
            )
            df.to_csv(csv_path, index=False)

            bboxes, annot_type = read_csv_annotation_file(csv_path, img_dir)
            assert annot_type == AnnotType.ORIGINAL

    def test_read_csv_persons(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "ann.csv")
            img_dir = os.path.join(tmpdir, "images")
            os.makedirs(img_dir)

            df = pd.DataFrame(
                {
                    "img_name": ["test.jpg"],
                    "face_x0": [0],
                    "face_y0": [0],
                    "face_x1": [100],
                    "face_y1": [100],
                    "person_x0": [0],
                    "person_y0": [0],
                    "person_x1": [200],
                    "person_y1": [200],
                    "age": [25],
                    "gender": ["M"],
                }
            )
            df.to_csv(csv_path, index=False)

            bboxes, annot_type = read_csv_annotation_file(csv_path, img_dir)
            assert annot_type == AnnotType.PERSONS


class TestPictureInfo:
    def test_has_face_bbox(self):
        info = PictureInfo("path", "25", "M", bbox=[0, 0, 100, 100])
        assert info.has_face_bbox is True

    def test_no_face_bbox(self):
        info = PictureInfo("path", "25", "M")
        assert info.has_face_bbox is False

    def test_has_gt_age_only(self):
        info = PictureInfo("path", "25", "-1")
        assert info.has_gt(only_age=True) is True

    def test_has_gt_both(self):
        info = PictureInfo("path", "-1", "-1")
        assert info.has_gt(only_age=False) is False

    def test_clear_bbox(self):
        info = PictureInfo("path", "25", "M", bbox=[0, 0, 100, 100])
        info.clear_face_bbox()
        assert info.bbox == [-1, -1, -1, -1]
