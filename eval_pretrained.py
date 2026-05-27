"""Evaluation script for pretrained Falcon models on benchmark datasets."""

from __future__ import annotations

import argparse
import json
import logging
from typing import Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from falcon.data.dataset import build as build_data
from falcon.eval_metrics import Metrics, time_sync, write_results
from falcon.model.inference import Falcon
from timm.utils import setup_default_logging

_logger = logging.getLogger("inference")
LOG_FREQUENCY = 10


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Falcon Validation")
    parser.add_argument("--dataset-images", required=True, type=str, help="path to images")
    parser.add_argument(
        "--dataset-annotations",
        required=True,
        type=str,
        help="path to annotations",
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        type=str,
        choices=["utk", "imdb", "lagenda", "fairface", "adience", "agedb", "cacd"],
    )
    parser.add_argument(
        "--split",
        default="validation",
        help="dataset splits separated by comma",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="path to falcon checkpoint",
    )
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--l-for-cs", type=int, default=5, help="L for cumulative score")
    parser.add_argument("--half", action="store_true", default=False)
    parser.add_argument("--with-persons", action="store_true", default=False)
    parser.add_argument("--disable-faces", action="store_true", default=False)
    parser.add_argument(
        "--draw-hist",
        action="store_true",
        help="Draw error histogram by age",
    )
    parser.add_argument(
        "--results-file",
        default="",
        type=str,
        help="Output csv/json file",
    )
    parser.add_argument(
        "--results-format",
        default="csv",
        type=str,
        choices=["csv", "json"],
    )
    return parser


def process_batch(
    falcon_model: Falcon,
    input: torch.Tensor,
    target: torch.Tensor,
    num_classes_gender: int = 2,
):
    """Run a single batch through the model and extract age/gender outputs."""
    start = time_sync()
    output = falcon_model.inference(input)
    assert not (all(target[:, 0] == -1) and all(target[:, 1] == -1))

    if falcon_model.meta.only_age:
        age_out = output
        gender_out, gender_target = None, None
    else:
        if falcon_model.config.num_classes > 3:
            gender_out = output[:, :2]
            age_out = output[:, 2:]
        else:
            gender_out = output[:, :num_classes_gender]
            age_out = output[:, num_classes_gender:]
        gender_target = target[:, 1]

    process_time = time_sync() - start
    age_target = target[:, 0:1]
    return age_out, age_target, gender_out, gender_target, process_time


def _filter_invalid_target(out: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mask = target != -1
    return out[mask], target[mask]


def postprocess_gender(gender_out: torch.Tensor, gender_target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if gender_target is None:
        return gender_out, gender_target
    return _filter_invalid_target(gender_out, gender_target)


def postprocess_age(age_out: torch.Tensor, age_target: torch.Tensor, dataset) -> Tuple[torch.Tensor, torch.Tensor]:
    """Denormalise age outputs and optionally map to classification bins."""
    age_out, age_target = _filter_invalid_target(age_out, age_target)

    if age_out.dim() > 1 and age_out.size(1) > 1 and dataset.age_classes is None:
        bins = torch.arange(age_out.size(1), device=age_out.device, dtype=torch.float32)
        age_out = (age_out.softmax(dim=-1) * bins).sum(dim=-1, keepdim=True)

    age_range = dataset.max_age - dataset.min_age
    age_out = age_out * age_range + dataset.min_age
    age_out = torch.clamp(age_out, min=0)

    if dataset.age_classes is not None:
        age_out = torch.round(age_out)
        if dataset._intervals.device != age_out.device:
            dataset._intervals = dataset._intervals.to(age_out.device)
        age_inds = torch.searchsorted(dataset._intervals, age_out, side="right") - 1
        age_out = age_inds
    else:
        age_target = age_target * age_range + dataset.min_age

    return age_out, age_target


def validate(args):
    """Run full validation and return a results dict."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    falcon_model = Falcon(
        args.checkpoint,
        args.device,
        half=args.half,
        use_persons=args.with_persons,
        disable_faces=args.disable_faces,
        verbose=True,
    )

    dataset, loader = build_data(
        name=args.dataset_name,
        images_path=args.dataset_images,
        annotations_path=args.dataset_annotations,
        split=args.split,
        model_config=falcon_model.config,
        workers=args.workers,
        batch_size=args.batch_size,
    )

    d_stat = Metrics(args.l_for_cs, args.draw_hist, dataset.age_classes)
    falcon_model.warmup(args.batch_size)

    preproc_end = time_sync()
    for batch_idx, (input, target) in enumerate(loader):
        preprocess_time = time_sync() - preproc_end
        age_out, age_target, gender_out, gender_target, proc_time = process_batch(
            falcon_model,
            input,
            target,
            dataset.num_classes_gender,
        )

        gender_out, gender_target = postprocess_gender(gender_out, gender_target)
        age_out, age_target = postprocess_age(age_out, age_target, dataset)

        d_stat.update_gender_accuracy(gender_out, gender_target)
        if d_stat.is_regression:
            d_stat.update_regression_age_metrics(age_out, age_target)
        else:
            d_stat.update_age_accuracy(age_out, age_target)
        d_stat.update_time(proc_time, preprocess_time, input.shape[0])

        if batch_idx % LOG_FREQUENCY == 0:
            _logger.info(f"Test: [{batch_idx:>4d}/{len(loader)}]  " f"{d_stat.get_info_str(input.size(0))}")
        preproc_end = time_sync()

    results = dict(
        model=args.checkpoint,
        dataset_name=args.dataset_name,
        param_count=round(falcon_model.param_count / 1e6, 2),
        img_size=falcon_model.input_size,
        use_faces=falcon_model.meta.use_face_crops,
        use_persons=falcon_model.meta.use_persons,
        in_chans=falcon_model.meta.in_chans,
        batch=args.batch_size,
    )
    results.update(d_stat.get_result())
    return results


def main():
    parser = get_parser()
    setup_default_logging()
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    results = validate(args)

    result_str = f" * Age Acc@1 {results['agetop1']:.3f} ({results['agetop1_err']:.3f})"
    if "gendertop1" in results:
        result_str += f" Gender Acc@1 {results['gendertop1']:.3f}" f" ({results['gendertop1_err']:.3f})"
    result_str += (
        f" Mean inference time {results['mean_inference_time']:.3f} ms "
        f"Mean preprocessing time {results['mean_preprocessing_time']:.3f}"
    )
    _logger.info(result_str)

    if args.draw_hist and "per_age_error" in results:
        err = [sum(v) / len(v) for v in results["per_age_error"].values()]
        ages = list(results["per_age_error"].keys())
        sns.scatterplot(x=ages, y=err, hue=err)
        plt.legend([], [], frameon=False)
        plt.xlabel("Age")
        plt.ylabel("MAE")
        plt.savefig("age_error.png", dpi=300)

    if args.results_file:
        write_results(args.results_file, results, format=args.results_format)

    print(f"--result\n{json.dumps(results, indent=4)}")


if __name__ == "__main__":
    main()
