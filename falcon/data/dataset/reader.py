from __future__ import annotations

import logging
import os
from functools import partial
from multiprocessing.pool import ThreadPool
from typing import Dict, List, Tuple

import cv2
import numpy as np
from falcon.data.io import AnnotType, PictureInfo, get_all_files, read_csv_annotation_file
from falcon.data.transforms import class_letterbox, iou
from timm.data.readers.reader import Reader
from tqdm import tqdm

CROP_ROUND_TOL = 0.3
MIN_PERSON_SIZE = 100
MIN_PERSON_CROP_AFTERCUT_RATIO = 0.4

_logger = logging.getLogger("ReaderAgeGender")


class ReaderAgeGender(Reader):
    def __init__(
        self,
        images_path,
        annotations_path,
        split="validation",
        target_size=224,
        min_size=5,
        seed=1234,
        with_persons=False,
        min_person_size=MIN_PERSON_SIZE,
        disable_faces=False,
        only_age=False,
        min_person_aftercut_ratio=MIN_PERSON_CROP_AFTERCUT_RATIO,
        crop_round_tol=CROP_ROUND_TOL,
    ):
        super().__init__()
        self.with_persons = with_persons
        self.disable_faces = disable_faces
        self.only_age = only_age
        self.crop_out_color = (0, 0, 0)
        self.empty_crop = np.full((target_size, target_size, 3), self.crop_out_color, dtype=np.uint8)
        self.min_person_size = min_person_size
        self.min_person_aftercut_ratio = min_person_aftercut_ratio
        self.crop_round_tol = crop_round_tol

        self.splits = [s.strip() for s in split.split(",") if s.strip()]
        assert self.splits, "Incorrect split arg"

        self.min_size = min_size
        self.seed = seed
        self.target_size = target_size

        self._ann: Dict[str, List[PictureInfo]] = {}
        self._associated_objects: Dict[str, Dict[int, List[List[int]]]] = {}
        self._faces_list: List[Tuple[str, int]] = []

        self._read_annotations(images_path, annotations_path)
        _logger.info(f"Dataset length: {len(self._faces_list)} crops")

    def __getitem__(self, index):
        return self._read_img_and_label(index)

    def __len__(self):
        return len(self._faces_list)

    def _filename(self, index, basename=False, absolute=False):
        img_p = self._faces_list[index][0]
        return os.path.basename(img_p) if basename else img_p

    def _read_annotations(self, images_path, csvs_path):
        self._ann = {}
        self._faces_list = []
        self._associated_objects = {}

        csvs = get_all_files(csvs_path, [".csv"])
        csvs = [
            c for c in csvs
            if any(split_name in os.path.basename(c) for split_name in self.splits)
        ]

        for csv in csvs:
            db, ann_type = read_csv_annotation_file(csv, images_path)
            if self.with_persons and ann_type != AnnotType.PERSONS:
                raise ValueError(
                    f"Annotation file {csv} has no persons, "
                    "but persons are required."
                )
            self._ann.update(db)

        if not self._ann:
            raise ValueError("Annotations are empty!")

        self._ann, self._associated_objects = self.prepare_annotations()
        for img_path in self._ann:
            for idx, sample in enumerate(self._ann[img_path]):
                assert sample.has_gt(self.only_age)
                self._faces_list.append((img_path, idx))

    def _read_img_and_label(self, index):
        img_p, face_index = self._faces_list[index]
        ann = self._ann[img_p][face_index]
        img = cv2.imread(img_p)

        face_empty = True
        if ann.has_face_bbox and not (self.with_persons and self.disable_faces):
            face_crop, face_empty = self._get_crop(ann.bbox, img)
        if not self.with_persons and face_empty:
            raise ValueError("Annotations must be checked by prepare_annotations")

        if face_empty:
            face_crop = self.empty_crop

        person_empty = True
        if self.with_persons or self.disable_faces:
            if ann.has_person_bbox:
                objects = self._associated_objects[img_p][face_index]
                person_crop, person_empty = self._get_crop(
                    ann.person_bbox, img,
                    crop_out_color=self.crop_out_color,
                    asced_objects=objects,
                )
            if face_empty and person_empty:
                raise ValueError("Annotations must be checked by prepare_annotations")

        if person_empty:
            person_crop = self.empty_crop

        return (face_crop, person_crop), [ann.age, ann.gender]

    def _get_crop(self, bbox, img, asced_objects=None, crop_out_color=(0, 0, 0)):
        xmin, ymin, xmax, ymax = bbox
        assert (ymax - ymin >= self.min_size) and (xmax - xmin >= self.min_size)

        crop = img[ymin:ymax, xmin:xmax]

        if asced_objects:
            crop, empty = _cropout_asced_objs(
                asced_objects, bbox, crop.copy(),
                crop_out_color=crop_out_color,
                min_person_size=self.min_person_size,
                crop_round_tol=self.crop_round_tol,
                min_person_aftercut_ratio=self.min_person_aftercut_ratio,
            )
            if empty:
                crop = self.empty_crop
        else:
            empty = False

        crop = class_letterbox(crop, new_shape=(self.target_size, self.target_size), color=crop_out_color)
        return crop, empty

    def prepare_annotations(self):
        good_anns: Dict[str, List[PictureInfo]] = {}
        all_associated_objects: Dict[str, Dict[int, List[List[int]]]] = {}

        if not self.with_persons:
            for img_path, bboxes in self._ann.items():
                for sample in bboxes:
                    sample.clear_person_bbox()

        verify = partial(
            verify_images,
            min_size=self.min_size,
            min_person_size=self.min_person_size,
            with_persons=self.with_persons,
            disable_faces=self.disable_faces,
            crop_round_tol=self.crop_round_tol,
            min_person_aftercut_ratio=self.min_person_aftercut_ratio,
            only_age=self.only_age,
        )
        num_threads = min(8, os.cpu_count())

        broken = 0
        skipped = 0
        total_skipped = 0

        with ThreadPool(num_threads) as pool:
            pbar = tqdm(
                pool.imap_unordered(verify, list(self._ann.items())),
                desc="Check annotations...",
                total=len(self._ann),
            )
            for img_info, associated_objects, msgs, is_corrupted, is_empty, skipped_crops in pbar:
                broken += int(is_corrupted)
                total_skipped += skipped_crops
                skipped += int(is_empty)
                if img_info is not None:
                    img_path, img_samples = img_info
                    good_anns[img_path] = img_samples
                    all_associated_objects[img_path] = associated_objects
                pbar.set_description(
                    f"Check annotations... {skipped} skipped ({total_skipped} crops), "
                    f"{broken} corrupted"
                )

        for msg in set(msgs if 'msgs' in dir() else []):
            pass
        print(f"\nLeft images: {len(good_anns)}")
        return good_anns, all_associated_objects


def verify_images(img_info, min_size, min_person_size, with_persons, disable_faces,
                  crop_round_tol, min_person_aftercut_ratio, only_age):
    img_path, img_samples = img_info
    msgs = []
    skipped_crops = 0
    is_corrupted = False

    try:
        im_cv = cv2.imread(img_path)
        im_h, im_w = im_cv.shape[:2]
    except Exception:
        return None, {}, [f"Can't load {img_path}"], True, False, 0

    def correct_bbox(bbox, min_sz, h, w):
        ymin, ymax, xmin, xmax = _clamp_bbox(bbox, h, w)
        if (ymax - ymin) < min_sz or (xmax - xmin) < min_sz:
            return False, [-1, -1, -1, -1]
        return True, [xmin, ymin, xmax, ymax]

    out_samples = []
    for sample in img_samples:
        if sample.has_face_bbox:
            ok, sample.bbox = correct_bbox(sample.bbox, min_size, im_h, im_w)
            if not ok and sample.has_gt(only_age):
                msgs.append("Small face. Passing..")
                skipped_crops += 1

        if sample.has_person_bbox:
            ok, sample.person_bbox = correct_bbox(
                sample.person_bbox, max(min_person_size, min_size), im_h, im_w,
            )
            if not ok and sample.has_gt(only_age):
                msgs.append(f"Small person {img_path}. Passing..")
                skipped_crops += 1

        if sample.has_face_bbox or sample.has_person_bbox:
            out_samples.append(sample)
        elif sample.has_gt(only_age):
            msgs.append("No face/body. Passing..")
            skipped_crops += 1

    out_samples.sort(key=lambda s: 0 if s.has_gt(only_age) else 1)
    associated = find_associated_objects(out_samples, only_age=only_age)
    out_samples, associated, skipped_crops = filter_bad_samples(
        out_samples, associated, im_cv, msgs, skipped_crops,
        min_person_size=min_person_size, disable_faces=disable_faces and with_persons,
        with_persons=with_persons, crop_round_tol=crop_round_tol,
        min_person_aftercut_ratio=min_person_aftercut_ratio, only_age=only_age,
    )

    out_info = (img_path, out_samples) if out_samples else None
    return out_info, associated, msgs, is_corrupted, (not out_samples), skipped_crops


def filter_bad_samples(out_samples, associated_objects, im_cv, msgs, skipped_crops, **kw):
    with_persons = kw["with_persons"]
    disable_faces = kw["disable_faces"]
    min_person_size = kw["min_person_size"]
    crop_round_tol = kw["crop_round_tol"]
    min_person_aftercut_ratio = kw["min_person_aftercut_ratio"]
    only_age = kw["only_age"]

    inds = [i for i, s in enumerate(out_samples) if s.has_gt(only_age)]
    out_samples, associated_objects = _filter_by_ind(out_samples, associated_objects, inds)

    if disable_faces:
        for s in out_samples:
            s.clear_face_bbox()
        inds = [i for i, s in enumerate(out_samples) if s.has_person_bbox]
        out_samples, associated_objects = _filter_by_ind(out_samples, associated_objects, inds)

    if with_persons or disable_faces:
        inds = []
        for i, sample in enumerate(out_samples):
            person_empty = True
            if sample.has_person_bbox:
                xmin, ymin, xmax, ymax = sample.person_bbox
                crop = im_cv[ymin:ymax, xmin:xmax]
                _, person_empty = _cropout_asced_objs(
                    associated_objects[i], sample.person_bbox, crop.copy(),
                    min_person_size=min_person_size, crop_round_tol=crop_round_tol,
                    min_person_aftercut_ratio=min_person_aftercut_ratio,
                )
            if person_empty and not sample.has_face_bbox:
                msgs.append("Small person after preprocessing. Passing..")
                skipped_crops += 1
            else:
                inds.append(i)
        out_samples, associated_objects = _filter_by_ind(out_samples, associated_objects, inds)

    return out_samples, associated_objects, skipped_crops


def _filter_by_ind(out_samples, associated_objects, inds):
    new_samples = []
    new_assoc = {}
    for new_idx, old_idx in enumerate(inds):
        new_assoc[new_idx] = associated_objects[old_idx]
        new_samples.append(out_samples[old_idx])
    return new_samples, new_assoc


def find_associated_objects(image_samples, iou_thresh=0.0001, only_age=False):
    result = {}
    for i, sample in enumerate(image_samples):
        result[i] = [sample.bbox] if sample.has_face_bbox else []
        if not sample.has_person_bbox or not sample.has_gt(only_age):
            continue
        ip_box = sample.person_bbox
        for j, other in enumerate(image_samples):
            if i == j:
                continue
            if other.has_face_bbox and _get_iou(other.bbox, ip_box) >= iou_thresh:
                result[i].append(other.bbox)
            if other.has_person_bbox and _get_iou(other.person_bbox, ip_box) >= iou_thresh:
                result[i].append(other.person_bbox)
    return result


def _cropout_asced_objs(asced_objects, person_bbox, crop, min_person_size,
                        crop_round_tol, min_person_aftercut_ratio, crop_out_color=(0, 0, 0)):
    xmin, ymin, xmax, ymax = person_bbox

    for aobj in asced_objects:
        ax1, ay1, ax2, ay2 = aobj
        ay1 = int(max(ay1 - ymin, 0))
        ax1 = int(max(ax1 - xmin, 0))
        ay2 = int(min(ay2 - ymin, ymax - ymin))
        ax2 = int(min(ax2 - xmin, xmax - xmin))
        crop[ay1:ay2, ax1:ax2] = crop_out_color

    remain = np.count_nonzero(crop) / (crop.shape[0] * crop.shape[1] * crop.shape[2])
    if (crop.shape[0] < min_person_size or crop.shape[1] < min_person_size) \
       or remain < min_person_aftercut_ratio:
        return None, True
    return crop, False


def _clamp_bbox(bbox, h, w):
    xmin, ymin, xmax, ymax = bbox
    return (
        min(max(ymin, 0), h),
        min(max(ymax, 0), h),
        min(max(xmin, 0), w),
        min(max(xmax, 0), w),
    )


def _get_iou(bbox1, bbox2):
    return iou(
        [bbox1[1], bbox1[0], bbox1[3], bbox1[2]],
        [bbox2[1], bbox2[0], bbox2[3], bbox2[2]],
    )
