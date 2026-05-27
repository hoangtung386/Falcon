from __future__ import annotations

from falcon.structures import PersonAndFaceCrops


class TestPersonAndFaceCrops:
    def test_empty_crops(self):
        crops = PersonAndFaceCrops()
        (bodies_inds, bodies), (faces_inds, faces) = crops.get_faces_with_bodies(
            use_persons=False,
            use_faces=True,
        )
        assert len(bodies) == 0
        assert len(faces) == 0
        assert len(bodies_inds) == 0
        assert len(faces_inds) == 0

    def test_faces_without_bodies(self):
        crops = PersonAndFaceCrops()
        crops.crops_faces_wo_body[0] = "fake_face"
        (bodies_inds, bodies), (faces_inds, faces) = crops.get_faces_with_bodies(
            use_persons=False,
            use_faces=True,
        )
        assert len(faces) == 1
        assert len(bodies) == 1  # None for face without body
        assert bodies[0] is None
