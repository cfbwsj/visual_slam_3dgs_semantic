import numpy as np
from scipy.spatial.transform import Rotation

from vslam.common import make_T, project, transform, backproject
from vslam.geometry.two_view import estimate_pair


def test_projection_roundtrip():
    # 投影后再按原深度反投影，应回到原始三维点。
    K = np.array([[500., 0, 320], [0, 500, 240], [0, 0, 1]])
    X = np.array([[0.2, -0.3, 2.], [-0.4, 0.5, 3.]])
    assert np.allclose(backproject(project(K, X), X[:, 2], K), X)

    # T 和 T 的逆变换连续作用后，应恢复原始点坐标。
    T = make_T(Rotation.from_rotvec([0.1, -0.05, 0.02]).as_matrix(), [0.4, 0.2, 0.1])
    assert np.allclose(transform(np.linalg.inv(T), transform(T, X)), X)


def test_two_view_known_scene():
    # 用已知三维点和相机运动生成两帧，验证恢复出的旋转和平移方向。
    rng = np.random.default_rng(2)
    X = rng.uniform([-1., -0.8, 3.], [1., 0.8, 6.], size=(150, 3))
    K = np.array([[500., 0, 320], [0, 500, 240], [0, 0, 1]])
    R = Rotation.from_rotvec([0.01, 0.04, -0.02]).as_matrix()
    t = np.array([-0.3, 0.01, 0.02])

    p0 = project(K, X)
    p1 = project(K, transform(make_T(R, t), X))
    result, report = estimate_pair(p0, p1, K, np.zeros(5))
    estimated = result["T_c1_c0"]

    angle = Rotation.from_matrix(estimated[:3, :3] @ R.T).magnitude()
    assert angle < 0.01
    assert np.dot(estimated[:3, 3], t / np.linalg.norm(t)) > 0.99
    assert report["median_reprojection_px"] < 0.1