from vslam.data.association import associate


def test_one_to_one_and_tolerance():
    # x 与 b 的时间差最小，因此这组对应应优先被选中。
    rgb = [(1.0, "a"), (1.01, "b"), (2.0, "c")]
    depth = [(1.009, "x"), (2.006, "y")]
    result = associate(rgb, depth, 0.02)

    assert len(result) == 2
    assert result[0][1] == "b"
    # 每张深度图只能出现一次，且所有时间差都在容差内。
    assert len({row[3] for row in result}) == len(result)
    assert all(abs(row[0] - row[2]) <= 0.02 for row in result)