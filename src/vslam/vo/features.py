import cv2                      # OpenCV：图像读写、ORB 特征、特征匹配
import numpy as np              # 数值数组：关键点坐标、描述子存储

def extract_orb(image_bgr, nfeatures=2000):
    # 转灰度：ORB 检测器只吃单通道图，彩色 3 通道会被 OpenCV 内部隐式转换但显式转更清晰
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # 构造 ORB 检测器；nfeatures 是"上限"不是"保证值"，弱纹理图可能远达不到
    orb = cv2.ORB_create(nfeatures=nfeatures)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    # 关键检查：图像模糊 / 纯色 / 弱纹理时 ORB 可能一个点都提不出来
    if descriptors is None:
        raise RuntimeError("ORB 未提取到任何描述子，请检查图像内容")

    return keypoints, descriptors


# ---------------------------------------------------------------------------
# 函数 2：match_ratio
# 作用：对两组 ORB 描述子做 KNN 匹配 + Lowe's ratio test，并按目标点去重
# 输入：descriptors0 —— 图 0 的描述子, (N0, 32) uint8
#       descriptors1 —— 图 1 的描述子, (N1, 32) uint8
#       ratio        —— 比值阈值，越小越严格（Lowe 推荐 0.75）
# 输出：matches —— list[cv2.DMatch]，按 distance 升序，且每个 trainIdx 只出现一次
# ---------------------------------------------------------------------------
def match_ratio(descriptors0, descriptors1, ratio=0.75):
    # BFMatcher = Brute Force，对每个 query 描述子和所有 train 描述子暴力算距离
    # NORM_HAMMING：ORB 描述子是二进制串，用汉明距离（数不同位的个数）才是正确的度量
    # crossCheck=False：关掉双向检查，因为我们要用 ratio test 来过滤，二者二选一即可
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # knnMatch(k=2)：为每个 query 描述子找最近的 2 个 train 描述子
    # 返回结构：list[list[DMatch]]，外层长度 = N0，内层通常是 2 个 DMatch
    # 注意：若 train 描述子总数 < 2，内层只有 1 个 —— 这就是下面要判 len 的原因
    pairs = matcher.knnMatch(descriptors0, descriptors1, k=2)

    # 用 dict 一遍完成"ratio 过滤 + 按 trainIdx 去重"
    # key = trainIdx（目标点编号），value = 目前找到的最优 DMatch
    # 用它而不是 list + set，是因为同 key 覆盖写天然保留最小值，省了排序
    best_by_train = {}

    for pair in pairs:
        # 防御接口约定：knnMatch 在 train 描述子不足 2 个时返回长度 1 的 pair
        # 不跳过的话下面 unpack 会直接 ValueError
        if len(pair) < 2:
            continue

        # 解构出最近邻和次近邻
        best, second = pair

        # Lowe's ratio test：
        #   最近邻距离 < ratio * 次近邻距离  →  说明这个匹配"显著优于第二选择"，可信
        #   否则说明最近邻也"模棱两可"（很像好几个目标点），判为歧义丢弃
        if best.distance >= ratio * second.distance:
            continue

        # 去重：同一个目标点可能被多个 query 匹配到，只保留距离最小的那条
        prev = best_by_train.get(best.trainIdx)
        if prev is None or best.distance < prev.distance:
            best_by_train[best.trainIdx] = best

    # 按距离升序返回：距离越小匹配越可信，
    # 下游 RANSAC / 三角化时质量高的匹配先入选，稳定性更好
    return sorted(best_by_train.values(), key=lambda m: m.distance)


# ---------------------------------------------------------------------------
# 函数 3：matched_points
# 作用：把 DMatch 里的索引翻译成实际像素坐标，方便喂给 OpenCV 几何函数
# 输入：keypoints0 —— 图 0 的关键点列表
#       keypoints1 —— 图 1 的关键点列表
#       matches    —— match_ratio 的输出
# 输出：(points0, points1)
#       两者均为 (N, 2) float64 数组，每行是 (x, y) 像素坐标，一一对应
# ---------------------------------------------------------------------------
def matched_points(keypoints0, keypoints1, matches):
    # m.queryIdx：该匹配在图 0 中对应的关键点下标
    # keypoints0[idx].pt：取关键点的像素坐标，是个 (x, y) 元组
    # 用列表推导把所有匹配的坐标收集起来，再整体转成 numpy 数组
    points0 = np.array(
        [keypoints0[m.queryIdx].pt for m in matches],
        dtype=np.float64,               # 显式 float64：cv2 几何函数要求浮点输入
    ).reshape(-1, 2)                    # 空列表时保证形状是 (0,2)，避免下游形状报错

    # 同理处理图 1 一侧
    points1 = np.array(
        [keypoints1[m.trainIdx].pt for m in matches],
        dtype=np.float64,
    ).reshape(-1, 2)

    # points0[i] 和 points1[i] 是同一条匹配在两帧中的像素位置，顺序严格对应
    return points0, points1