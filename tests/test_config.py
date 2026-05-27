from __future__ import annotations

from falcon.config import ModelConfig


class TestModelConfig:
    def test_default_values(self):
        cfg = ModelConfig()
        assert cfg.input_size == 224
        assert cfg.num_classes == 3
        assert cfg.age_head_mode == "regression"
        assert cfg.use_person_crops is False

    def test_face_crops_property(self):
        cfg = ModelConfig()
        assert cfg.use_face_crops is True

        cfg_no_face = ModelConfig(disable_faces=True, with_persons_model=True)
        assert cfg_no_face.use_face_crops is False

    def test_person_crops_property(self):
        cfg = ModelConfig()
        assert cfg.use_person_crops is False

        cfg_with = ModelConfig(with_persons_model=True, use_persons=True)
        assert cfg_with.use_person_crops is True
