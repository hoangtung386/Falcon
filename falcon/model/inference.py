"""Inference wrapper for Falcon age/gender models.

Handles checkpoint loading, model configuration, image preprocessing,
and filling results into detection structures.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from falcon.config import ModelConfig
from falcon.data.transforms import prepare_classification_images
from falcon.model.factory import create_model
from falcon.structures import PersonAndFaceResult
from timm.data import resolve_data_config

_logger = logging.getLogger("Falcon")
_has_compile = hasattr(torch, "compile")

__all__ = ["Meta", "Falcon"]


class Meta:
    """Container for model metadata extracted from a checkpoint.

    Attributes:
        min_age: Minimum age encountered during training.
        max_age: Maximum age encountered during training.
        avg_age: Average age (midpoint) used for normalisation.
        num_classes: Number of output classes.
        in_chans: Number of input channels (3 or 6).
        with_persons_model: Whether the model handles face+body inputs.
        disable_faces: Whether face crops are disabled.
        use_persons: Whether person crops are enabled.
        only_age: Whether the model skips gender prediction.
        num_classes_gender: Number of gender classes.
        input_size: Spatial input size.
    """

    def __init__(self):
        self.min_age: float | None = None
        self.max_age: float | None = None
        self.avg_age: float | None = None
        self.num_classes: int | None = None
        self.in_chans: int = 3
        self.with_persons_model: bool = False
        self.disable_faces: bool = False
        self.use_persons: bool = True
        self.only_age: bool = False
        self.num_classes_gender: int = 2
        self.input_size: int = 224

    def load_from_ckpt(
        self,
        ckpt_path: str,
        disable_faces: bool = False,
        use_persons: bool = True,
    ) -> "Meta":
        """Populate metadata by inspecting a saved checkpoint.

        Args:
            ckpt_path: Path to the checkpoint file.
            disable_faces: Whether face crops are disabled.
            use_persons: Whether person crops are enabled.

        Returns:
            Self for chaining.
        """
        state = torch.load(ckpt_path, map_location="cpu")

        self.min_age = state.get("min_age", 0.0)
        self.max_age = state.get("max_age", 100.0)
        self.avg_age = state.get("avg_age", 50.0)
        self.only_age = state.get("no_gender", False)
        self.disable_faces = disable_faces

        self.with_persons_model = state.get("with_persons_model", False)
        if not self.with_persons_model:
            sd = state.get("state_dict", {})
            self.with_persons_model = "patch_embed.conv1.0.weight" in sd

        self.num_classes = 1 if self.only_age else 3
        self.in_chans = 3 if not self.with_persons_model else 6
        self.use_persons = use_persons and self.with_persons_model

        if not self.with_persons_model and self.disable_faces:
            raise ValueError("disable-faces requires with-persons model")
        if self.with_persons_model and self.disable_faces and not self.use_persons:
            raise ValueError("Cannot disable both faces and persons. " "Set --with-persons to run with --disable-faces")

        self.input_size = state["state_dict"]["pos_embed"].shape[1] * 16
        return self

    def __str__(self):
        attrs = vars(self)
        attrs.update(
            {
                "use_person_crops": self.use_person_crops,
                "use_face_crops": self.use_face_crops,
            }
        )
        return ", ".join(f"{k}: {v}" for k, v in attrs.items())

    @property
    def use_person_crops(self) -> bool:
        """Whether person crops are active."""
        return self.with_persons_model and self.use_persons

    @property
    def use_face_crops(self) -> bool:
        """Whether face crops are active."""
        return not self.disable_faces or not self.with_persons_model


class Falcon:
    """Ready-to-use Falcon age/gender inference model.

    Args:
        ckpt_path: Path to the trained checkpoint.
        device: Device string (``"cuda"``, ``"cpu"``).
        half: Whether to run in FP16.
        disable_faces: Skip face crops.
        use_persons: Enable person crops.
        verbose: Enable verbose logging.
        torchcompile: TorchDynamo backend (e.g. ``"inductor"``).
    """

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        half: bool = True,
        disable_faces: bool = False,
        use_persons: bool = True,
        verbose: bool = False,
        torchcompile: Optional[str] = None,
    ):
        self.verbose = verbose
        self.device = torch.device(device)
        self.half = half and self.device.type != "cpu"

        self.meta = Meta().load_from_ckpt(ckpt_path, disable_faces, use_persons)
        if self.verbose:
            _logger.info(f"Model meta:\n{str(self.meta)}")

        model_name = f"falcon_d1_{self.meta.input_size}"
        self.model = create_model(
            model_name=model_name,
            num_classes=self.meta.num_classes,
            in_chans=self.meta.in_chans,
            pretrained=False,
            checkpoint_path=ckpt_path,
            filter_keys=["fds."],
        )
        self.param_count = sum(m.numel() for m in self.model.parameters())
        _logger.info(f"Model {model_name} created, param count: {self.param_count}")

        self.data_config = resolve_data_config(
            model=self.model,
            verbose=verbose,
            use_test_size=True,
        )
        self.data_config["crop_pct"] = 1.0
        c, h, w = self.data_config["input_size"]
        assert h == w, "Incorrect data_config"
        self.input_size = w

        self.model = self.model.to(self.device)

        if torchcompile:
            assert _has_compile, "torch.compile() requires PyTorch 2.0+"
            torch._dynamo.reset()
            self.model = torch.compile(self.model, backend=torchcompile)

        self.model.eval()
        if self.half:
            self.model = self.model.half()

        self.config = ModelConfig(
            input_size=self.input_size,
            max_age=self.meta.max_age,
            min_age=self.meta.min_age,
            avg_age=self.meta.avg_age,
            with_persons_model=self.meta.with_persons_model,
            use_persons=self.meta.use_persons,
            disable_faces=self.meta.disable_faces,
            only_age=self.meta.only_age,
            num_classes=self.meta.num_classes,
            in_chans=self.meta.in_chans,
            mean=self.data_config["mean"],
            std=self.data_config["std"],
            crop_pct=self.data_config["crop_pct"],
            crop_mode=self.data_config["crop_mode"],
            device=self.device,
        )

    def warmup(self, batch_size: int, steps: int = 10):
        """Run dummy inference to warm up the GPU/compiler.

        Args:
            batch_size: Batch size for the dummy tensor.
            steps: Number of warmup iterations.
        """
        if self.meta.with_persons_model:
            input_size = (6, self.input_size, self.input_size)
        else:
            input_size = self.data_config["input_size"]

        dummy = torch.randn((batch_size,) + tuple(input_size)).to(self.device)
        for _ in range(steps):
            self.inference(dummy)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @torch.no_grad()
    def inference(self, model_input: torch.Tensor) -> torch.Tensor:
        """Run the model on a preprocessed input tensor.

        Args:
            model_input: Normalised tensor of shape ``(B, C, H, W)``.

        Returns:
            Raw model output tensor.
        """
        if self.half:
            model_input = model_input.half()
        return self.model(model_input)

    def predict(self, image: np.ndarray, detected_bboxes: PersonAndFaceResult):
        """Detect age/gender for all objects in *detected_bboxes*.

        Args:
            image: Full BGR image.
            detected_bboxes: Detection result from YOLO.
        """
        should_skip = (
            detected_bboxes.n_objects == 0
            or (not self.meta.use_persons and detected_bboxes.n_faces == 0)
            or (self.meta.disable_faces and detected_bboxes.n_persons == 0)
        )
        if should_skip:
            return

        faces_input, person_input, faces_inds, bodies_inds = self.prepare_crops(
            image,
            detected_bboxes,
        )
        if faces_input is None and person_input is None:
            return

        if self.meta.with_persons_model:
            model_input = torch.cat((faces_input, person_input), dim=1)
        else:
            model_input = faces_input
        output = self.inference(model_input)
        self.fill_in_results(output, detected_bboxes, faces_inds, bodies_inds)

    def fill_in_results(self, output, detected_bboxes, faces_inds, bodies_inds):
        """Parse model output and assign age/gender to detection boxes."""
        has_distribution_head = hasattr(self.model, "age_head") and self.model.age_head is not None

        if has_distribution_head:
            age_bins = self.model.age_bins.to(output.device)
            age_logits = output[:, 2:]
            age_probs = age_logits.softmax(dim=-1)
            age_tensor = (age_probs * age_bins).sum(dim=-1)
            gender_out = output[:, :2]
            gender_probs, gender_indx = gender_out.softmax(-1).topk(1)
        elif self.meta.only_age:
            age_tensor = output.squeeze(-1)
            gender_probs, gender_indx = None, None
        else:
            age_tensor = output[:, 2]
            gender_out = output[:, :2].softmax(-1)
            gender_probs, gender_indx = gender_out.topk(1)

        assert output.shape[0] == len(faces_inds) == len(bodies_inds)

        range_age = self.meta.max_age - self.meta.min_age
        for i in range(output.shape[0]):
            face_ind = faces_inds[i]
            body_ind = bodies_inds[i]

            age_val = age_tensor[i].item()
            if not has_distribution_head:
                age_val = age_val * range_age + self.meta.min_age
            age_val = round(age_val, 2)

            detected_bboxes.set_age(face_ind, age_val)
            detected_bboxes.set_age(body_ind, age_val)
            _logger.info(f"\tage: {age_val}")

            if gender_probs is not None:
                gender = "male" if gender_indx[i].item() == 0 else "female"
                score = gender_probs[i].item()
                _logger.info(f"\tgender: {gender} [{int(score * 100)}%]")
                detected_bboxes.set_gender(face_ind, gender, score)
                detected_bboxes.set_gender(body_ind, gender, score)

    def prepare_crops(self, image, detected_bboxes):
        """Extract and preprocess face/person crops from the image."""
        if self.meta.use_person_crops and self.meta.use_face_crops:
            detected_bboxes.associate_faces_with_persons()

        crops = detected_bboxes.collect_crops(image)
        (bodies_inds, bodies_crops), (
            faces_inds,
            faces_crops,
        ) = crops.get_faces_with_bodies(
            self.meta.use_person_crops,
            self.meta.use_face_crops,
        )

        faces_input = prepare_classification_images(
            faces_crops,
            self.input_size,
            self.data_config["mean"],
            self.data_config["std"],
            device=self.device,
        )
        person_input = prepare_classification_images(
            bodies_crops,
            self.input_size,
            self.data_config["mean"],
            self.data_config["std"],
            device=self.device,
        )

        _logger.info(
            f"faces_input: {faces_input.shape if faces_input is not None else None}, "
            f"person_input: {person_input.shape if person_input is not None else None}"
        )
        return faces_input, person_input, faces_inds, bodies_inds
