from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
import tyro

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset


# ============================================================
# 1. 数据集配置
# ============================================================
# 这里只保留当前真正需要的映射：
# - 图像命名规则
# - qpos 列名
# - gripper 列名
#
# 不再保留 eef/state_map，因为你已经明确不需要 eef 数据。
# ============================================================

DATASET_CONFIG: dict[str, Any] = {
    "meta": {
        "frame_id_col": "frame_id",
        "task": "insert usb into red hole",
        "robot_type": "realman",
        "fps": 20,
    },

    "images": {
        "image": {
            "prefix": "RealSense_head_rgb_",
            "ext": ".jpg",
        },
        "wrist_image": {
            "prefix": "RealSense_wrist_rgb_",
            "ext": ".jpg",
        },
    },

    # 当前真正需要的 state 定义：
    # 左臂7关节 + 左夹爪1维 + 右臂7关节 + 右夹爪1维 = 16维
    "eef_map": {
        "eef": [
            "right_arm_state_0",
            "right_arm_state_1",
            "right_arm_state_2",
            "right_arm_state_3",
            "right_arm_state_4",
            "right_arm_state_5",
        ],
    },

    "gripper_map": {
        "gripper": "right_arm_gripper",
    },
}


# ============================================================
# 2. 工具函数
# ============================================================

def load_rgb_image(image_path: Path) -> np.ndarray:
    """读取 RGB 图像，返回 HWC 格式 numpy 数组。"""
    return np.array(Image.open(image_path).convert("RGB"))


def validate_required_columns(df: pd.DataFrame, config: dict[str, Any]) -> None:
    """检查 xlsx 是否包含当前配置需要的所有列。"""
    required_cols: list[str] = []

    for cols in config["eef_map"].values():
        required_cols.extend(cols)

    required_cols.extend(config["gripper_map"].values())
    required_cols.append(config["meta"]["frame_id_col"])

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"xlsx 缺少以下列：{missing}")


def build_robot_state_sequence(df: pd.DataFrame, config: dict[str, Any]) -> np.ndarray:
    """
    构造机器人状态序列。

    最终 state 定义为：
        [单臂6维EEF, 夹爪1维]

    因此每一帧 state 维度为 7。
    """
    eef = df[config["eef_map"]["eef"]].to_numpy(dtype=np.float32)

    gripper = df[[config["gripper_map"]["gripper"]]].to_numpy(dtype=np.float32)

    state_seq = np.concatenate(
        [eef, gripper],
        axis=1,
    )
    return state_seq


def build_state_action_pairs(sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    用时序错位法构造 state-actions 对。

    规则：
        state_t    = sequence[t]
        actions_t  = sequence[t+1]

    输入：
        sequence: [T, D]

    输出：
        states:  [T-1, D]
        actions: [T-1, D]
    """
    if len(sequence) < 2:
        raise ValueError("帧数少于 2,无法构造 state-actions 对。")

    states = sequence[:-1]
    actions = sequence[1:]
    return states, actions


def build_image_paths(
    df: pd.DataFrame,
    episode_dir: Path,
    config: dict[str, Any],
) -> dict[str, list[Path]]:
    """
    根据 frame_id 和配置中的文件命名规则，构造每一帧对应的图像路径。
    """
    frame_id_col = config["meta"]["frame_id_col"]
    frame_ids = df[frame_id_col].astype(int).tolist()

    image_paths: dict[str, list[Path]] = {}

    for output_key, rule in config["images"].items():
        paths = []
        for frame_id in frame_ids:
            image_path = episode_dir / f"{rule['prefix']}{frame_id}{rule['ext']}"
            if not image_path.exists():
                raise FileNotFoundError(f"找不到图像文件：{image_path}")
            paths.append(image_path)
        image_paths[output_key] = paths

    return image_paths


def infer_image_shape_from_episode(
    episode_dir: Path,
    config: dict[str, Any],
) -> dict[str, tuple[int, int, int]]:
    """
    从某个 episode 中自动推断图像的 shape。
    """
    xlsx_path = episode_dir / "episode_data.xlsx"
    if not xlsx_path.exists():
        raise FileNotFoundError(f"找不到 xlsx 文件：{xlsx_path}")

    df = pd.read_excel(xlsx_path)
    validate_required_columns(df, config)

    first_frame_id = int(df.iloc[0][config["meta"]["frame_id_col"]])

    shapes: dict[str, tuple[int, int, int]] = {}
    for output_key, rule in config["images"].items():
        image_path = episode_dir / f"{rule['prefix']}{first_frame_id}{rule['ext']}"
        image = load_rgb_image(image_path)
        shapes[output_key] = tuple(image.shape)

    return shapes


def iter_aligned_frames(episode_dir: Path, config: dict[str, Any]):
    """
    读取一个 episode，生成可直接写入 LeRobot 的逐帧样本。

    对齐规则：
        原始共有 T 帧机器人状态：
            s0, s1, ..., s_{T-1}

        最终构造：
            state   = s0, s1, ..., s_{T-2}
            actions = s1, s2, ..., s_{T-1}

        当前帧图像使用：
            img0, img1, ..., img_{T-2}

    因此最终有效样本数为 T-1。
    """
    xlsx_path = episode_dir / "episode_data.xlsx"
    if not xlsx_path.exists():
        raise FileNotFoundError(f"找不到 xlsx 文件：{xlsx_path}")

    df = pd.read_excel(xlsx_path)
    validate_required_columns(df, config)

    # 构造基于 qpos + gripper 的 16维状态序列
    state_seq = build_robot_state_sequence(df, config)

    # 用时序错位法构造 state/actions
    states, actions = build_state_action_pairs(state_seq)

    # 构造图像路径
    image_paths = build_image_paths(df, episode_dir, config)

    num_samples = len(states)

    for i in range(num_samples):
        frame: dict[str, Any] = {
            "task": config["meta"]["task"],
            "state": states[i],
            "actions": actions[i],
        }

        for image_key, paths in image_paths.items():
            frame[image_key] = load_rgb_image(paths[i])

        yield frame


def find_episode_dirs(data_root: Path, include: list[str] | None = None) -> list[Path]:
    """
    在 data_root 下递归查找所有包含 episode_data.xlsx 的 episode 目录。
    include: 需要包含的父目录名称列表（如 ["insert_usb_left_success", "insert_usb_right"]）。
             为 None 时包含所有。
    """
    episode_dirs = []
    for xlsx_path in sorted(data_root.rglob("episode_data.xlsx")):
        episode_dir = xlsx_path.parent
        if include is not None and not any(inc in episode_dir.parts for inc in include):
            continue
        episode_dirs.append(episode_dir)

    if not episode_dirs:
        raise FileNotFoundError(
            f"在 {data_root} 下没有找到包含 episode_data.xlsx 的 episode 子目录。"
        )

    return episode_dirs


def create_lerobot_features(
    image_shapes: dict[str, tuple[int, int, int]],
) -> dict[str, Any]:
    """
    根据图像 shape 构造 LeRobot 的 features 定义。
    """
    features: dict[str, Any] = {}

    for image_key, shape in image_shapes.items():
        features[image_key] = {
            "dtype": "image",
            "shape": shape,
            "names": ["height", "width", "channel"],
        }

    # 7维：
    # 单臂7关节 + 夹爪1维
    features["state"] = {
        "dtype": "float32",
        "shape": (7,),
        "names": ["state"],
    }

    features["actions"] = {
        "dtype": "float32",
        "shape": (7,),
        "names": ["actions"],
    }

    return features


# ============================================================
# 3. 主逻辑
# ============================================================

def main(
    data_dir: str,
    repo_name: str = "",
    include: list[str] | None = None,
    push_to_hub: bool = False,
    clear_output: bool = True,
    resume: bool = False,
    skip_log: str = "skip_episodes.log",
):
    """
    把自定义 episode 数据转换成 LeRobot 数据集。支持断点续传。

    输入目录结构示例：
        data_dir/
          group_1/
            episode_0001/
              episode_data.xlsx
              RealSense_head_rgb_0.png
              RealSense_right_wrist_rgb_0.png
              ...
          group_2/
            episode_0002/
              episode_data.xlsx
              ...

    输出位置：
        HF_LEROBOT_HOME / repo_name

    resume: True 时跳过已完成和已跳过的 episode，追加未处理的。
    include: 需要转换的子目录名称列表。为 None 时转换所有。
    """
    import datetime

    data_root = Path(data_dir)
    if not data_root.exists():
        raise FileNotFoundError(f"数据目录不存在：{data_root}")

    if repo_name == "":
        raise ValueError(f"Lerobot数据集名臣未定义:{repo_name}")

    episode_dirs = find_episode_dirs(data_root, include=include)

    # LeRobot 最终本地保存路径
    output_path = HF_LEROBOT_HOME / repo_name

    # --- 断点续传：读取已完成/已跳过的 episode 记录 ---
    DONE_FILE = output_path / ".done_episodes.txt"
    SKIP_FILE = output_path / ".skip_episodes.txt"
    done_set: set[str] = set()
    skip_set: set[str] = set()

    if resume and output_path.exists():
        if DONE_FILE.exists():
            done_set = set(DONE_FILE.read_text().strip().split("\n")) - {""}
            print(f"[INFO] 续传: 已加载 {len(done_set)} 个已完成 episode 记录")
        if SKIP_FILE.exists():
            skip_set = set(SKIP_FILE.read_text().strip().split("\n")) - {""}
            print(f"[INFO] 续传: 已加载 {len(skip_set)} 个已跳过 episode 记录")

    # 过滤掉已完成/已跳过的 episode
    pending_dirs = [d for d in episode_dirs if str(d) not in done_set and str(d) not in skip_set]
    skipped_episodes: list[tuple[str, str]] = []

    if resume and pending_dirs:
        print(f"[INFO] 续传: 剩余 {len(pending_dirs)} 个 episode 待处理")

    if not pending_dirs:
        print("[INFO] 所有 episode 已处理完毕，无需续传。")
        return

    # 自动推断图像尺寸（从待处理的 episode 中找第一个可用的）
    image_shapes = None
    for ep_dir in episode_dirs:  # 从全部 episode 中推断 shape（不限于 pending）
        try:
            image_shapes = infer_image_shape_from_episode(ep_dir, DATASET_CONFIG)
            break
        except Exception as e:
            print(f"[WARN] 跳过损坏的 episode (用于推断shape): {ep_dir}, 错误: {e}")
    if image_shapes is None:
        raise RuntimeError("无法从任何 episode 推断图像尺寸，所有 episode 可能都已损坏")

    # 构造 features
    features = create_lerobot_features(image_shapes)

    # 如果需要（非续传且 clear_output），先删除旧输出
    if clear_output and output_path.exists() and not resume:
        shutil.rmtree(output_path)

    print(f"[INFO] HF_LEROBOT_HOME = {HF_LEROBOT_HOME}")
    print(f"[INFO] 输出目录 = {output_path}")

    # 创建或打开已有数据集
    if resume and output_path.exists():
        print("[INFO] 续传: 打开已有数据集...")
        # LeRobotDataset 直接传入 repo_id 会打开已有数据集
        dataset = LeRobotDataset(
            repo_id=repo_name,
        )
        # 开启多线程图像编码，否则续传时写入速度会很慢
        dataset.start_image_writer(
            num_processes=4,
            num_threads=8,
        )
        # 从已有数据集统计已完成的 episode 数和帧数（用于最终报告）
        prev_episodes = len(done_set)
        prev_frames = dataset.num_frames
        print(f"[INFO] 续传: 已有 {prev_episodes} episodes, {prev_frames} frames")
    else:
        dataset = LeRobotDataset.create(
            repo_id=repo_name,
            robot_type=DATASET_CONFIG["meta"]["robot_type"],
            fps=DATASET_CONFIG["meta"]["fps"],
            features=features,
            image_writer_threads=8,
            image_writer_processes=4,
        )
        prev_episodes = 0
        prev_frames = 0

    total_frames = 0
    total_episodes = 0

    # 确保输出目录存在
    output_path.mkdir(parents=True, exist_ok=True)

    for episode_dir in pending_dirs:
        try:
            print(f"[INFO] 正在转换 episode: {episode_dir}")

            frame_count_this_episode = 0
            for frame in iter_aligned_frames(episode_dir, DATASET_CONFIG):
                dataset.add_frame(frame)
                frame_count_this_episode += 1

            dataset.save_episode()

            total_frames += frame_count_this_episode
            total_episodes += 1

            # 立即记录到 done 文件
            with open(DONE_FILE, "a") as f:
                f.write(f"{episode_dir}\n")

            print(
                f"[INFO] episode 转换完成: {episode_dir.name}, "
                f"有效样本数 = {frame_count_this_episode}"
            )
        except Exception as e:
            error_msg = f"{e.__class__.__name__}: {e}"
            skipped_episodes.append((str(episode_dir), error_msg))
            # 记录跳过
            with open(SKIP_FILE, "a") as f:
                f.write(f"{episode_dir}\n")
            print(f"[SKIP] 跳过损坏的 episode: {episode_dir}, 错误: {error_msg}")

    # 写入完整的跳过日志
    log_path = output_path / skip_log
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"# 跳过的 episode 日志\n")
        f.write(f"# 生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# 本轮跳过: {len(skipped_episodes)}\n")
        f.write(f"# 累计跳过: {len(skip_set) + len(skipped_episodes)}\n\n")
        for ep_path, err in skipped_episodes:
            f.write(f"[SKIP] {ep_path}\n  错误: {err}\n\n")

    print(f"[INFO] 本轮转换: {total_episodes} episodes, {total_frames} frames")
    print(f"[INFO] 累计: {prev_episodes + total_episodes} episodes, {prev_frames + total_frames} frames")
    print(f"[INFO] 跳过的 episode 数 = {len(skipped_episodes)}")
    if skipped_episodes:
        print(f"[INFO] 跳过日志已保存到: {log_path}")
    print(f"[INFO] 数据已保存到 = {output_path}")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["custom", "single-arm","eef", "robot", "vision"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
        print("[INFO] 已上传到 Hugging Face Hub")


if __name__ == "__main__":
    tyro.cli(main)