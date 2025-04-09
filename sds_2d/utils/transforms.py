from kornia.geometry.conversions import (
    QuaternionCoeffOrder,
    quaternion_to_rotation_matrix,
)

from .typings import TensorType


def quats2rotmat_batched(quats: TensorType["N", 4]) -> TensorType["N", 3, 3]:
    return quaternion_to_rotation_matrix(quats, QuaternionCoeffOrder.WXYZ)  # noqa
