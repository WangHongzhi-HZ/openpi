from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import tyro

# Ensure repo root is importable when this file is run as a script.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.realman.device.arm_controller.RealMan import (
    RealManController as DeviceRealManController,
)
from examples.realman.device.camera.CameraManager import CAMERA_SERIALS
from examples.realman.device.camera.CameraManager import CameraManager
from openpi_client import image_tools
from openpi_client import websocket_client_policy

# START_POSITION_ANGLE_LEFT_ARM = [
#     4.566,
#     -25.713,
#     45.92,
#     -60.867,
#     -38.556,
#     -93.755,
#     31.807,
# ]

START_POSITION_ANGLE_LEFT_ARM = [
    18.814, 21.793, -8.455, 83.832, -3.093, 71.779, 287.604
]

START_POSITION_ANGLE_RIGHT_ARM = [
    -43.661,
    6.833,
    15.838,
    66.452,
    13.482,
    90.01,
    162.846,
]


# =========================
# 1. 参数定义
# =========================
@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 11257
    api_key: str | None = None

    num_steps: int = 200
    control_hz: float = 10.0

    # 图像输入尺寸，通常 pi0/pi0.5 预训练常用 224
    image_height: int = 224
    image_width: int = 224

    # 任务文本
    prompt: str = "stack cubes"

    # 是否启用双腕相机
    use_left_wrist: bool = True
    use_right_wrist: bool = True

    # 调试
    verbose: bool = True

    # RealSense 序列号（默认复用 CameraManager 里的配置）
    camera_type: str = "RealSense"
    main_camera_serial: str = CAMERA_SERIALS["RealSense"]["head"]
    left_wrist_camera_serial: str = CAMERA_SERIALS["RealSense"]["left_wrist"]
    right_wrist_camera_serial: str = CAMERA_SERIALS["RealSense"]["right_wrist"]

    # 双臂 IP
    left_arm_ip: str = "192.168.0.17"
    right_arm_ip: str = "192.168.0.19"

    # 夹爪动作限幅（训练数据分布约在 0~1000）
    gripper_min: float = 0.0
    gripper_max: float = 1000.0


# =========================
# 2. 真机接口封装
#    这里保留真实相机与机械臂状态接口
#    你后续只需要把 TODO 部分替换成自己的实现
# =========================
class RealCameraInterface:
    """
    真实相机接口占位类。
    你可以在这里接：
    - 顶视 RGB 相机
    - 左腕 RGB 相机
    - 右腕 RGB 相机

    返回格式统一为 HWC, uint8 / 或可转换为 uint8 的 ndarray
    """

    def __init__(self, args: Args) -> None:
        self._main_key = f"{args.camera_type}_head"
        self._left_key = f"{args.camera_type}_left_wrist"
        self._right_key = f"{args.camera_type}_right_wrist"

        camera_configs = [
            {
                "type": args.camera_type,
                "position": "head",
                "serial": args.main_camera_serial,
            }
        ]
        if args.use_left_wrist:
            camera_configs.append(
                {
                    "type": args.camera_type,
                    "position": "left_wrist",
                    "serial": args.left_wrist_camera_serial,
                }
            )
        if args.use_right_wrist:
            camera_configs.append(
                {
                    "type": args.camera_type,
                    "position": "right_wrist",
                    "serial": args.right_wrist_camera_serial,
                }
            )

        self._manager = CameraManager(camera_configs)
        if self._manager.res != 0:
            raise RuntimeError(
                "Failed to initialize all configured cameras. "
                "Please check camera serials and USB connection."
            )

    @staticmethod
    def _as_rgb_uint8(frame: np.ndarray) -> np.ndarray:
        frame = image_tools.convert_to_uint8(np.asarray(frame))
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f"Expected HWC RGB/BGR image, got shape={frame.shape}")
        # RealSense 返回 BGR，这里统一转成 RGB。
        return frame[..., ::-1].copy()

    @staticmethod
    def _extract_rgb(
        frames: dict[str, tuple[np.ndarray, np.ndarray, float]],
        key: str,
        camera_name: str,
    ) -> np.ndarray:
        if key not in frames:
            raise KeyError(f"Missing frame for {camera_name}: key='{key}'")
        rgb, _depth, _frame_time = frames[key]
        if rgb is None:
            raise RuntimeError(f"Received empty RGB frame from {camera_name}")
        return RealCameraInterface._as_rgb_uint8(rgb)

    def get_images(
        self,
        need_left_wrist: bool,
        need_right_wrist: bool,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        frames = self._manager.get_frames()
        main_img = self._extract_rgb(frames, self._main_key, "main camera")

        left_img = None
        if need_left_wrist:
            left_img = self._extract_rgb(frames, self._left_key, "left wrist camera")

        right_img = None
        if need_right_wrist:
            right_img = self._extract_rgb(frames, self._right_key, "right wrist camera")

        return main_img, left_img, right_img

    def get_main_image(self) -> np.ndarray:
        main_img, _left_img, _right_img = self.get_images(
            need_left_wrist=False,
            need_right_wrist=False,
        )
        return main_img

    def get_left_wrist_image(self) -> np.ndarray:
        _main_img, left_img, _right_img = self.get_images(
            need_left_wrist=True,
            need_right_wrist=False,
        )
        assert left_img is not None
        return left_img

    def get_right_wrist_image(self) -> np.ndarray:
        _main_img, _left_img, right_img = self.get_images(
            need_left_wrist=False,
            need_right_wrist=True,
        )
        assert right_img is not None
        return right_img

    def close(self) -> None:
        for camera in self._manager.get_cameras().values():
            try:
                camera.disconnect()
            except Exception as exc:  # noqa: BLE001
                logging.warning("Failed to disconnect camera: %s", exc)


class RealManStateInterface:
    """
    机械臂状态接口占位类。
    你后续在这里接入 rm_get_current_arm_state()。
    """

    def __init__(self, args: Args) -> None:
        self._left_arm = DeviceRealManController(args.left_arm_ip)
        self._right_arm = DeviceRealManController(args.right_arm_ip)
        self._gripper_min = float(args.gripper_min)
        self._gripper_max = float(args.gripper_max)

    def get_robot_state(self) -> np.ndarray:
        """
        获取策略需要的 state 向量。

        你给的示例里 state 是 shape=(16,)
        因此这里默认返回 16 维。
        一般可以来自：
        - 左臂 joint / eef
        - 右臂 joint / eef
        - gripper 状态
        - 其他 proprioception

        返回:
            np.ndarray, shape=(16,), dtype=float32
        """
        _left_eef, left_joint, left_gripper, _left_time = self._left_arm.get_state()
        _right_eef, right_joint, right_gripper, _right_time = self._right_arm.get_state()

        state = np.concatenate(
            [
                np.asarray(left_joint, dtype=np.float32),
                np.asarray([left_gripper], dtype=np.float32),
                np.asarray(right_joint, dtype=np.float32),
                np.asarray([right_gripper], dtype=np.float32),
            ],
            axis=0,
        )
        if state.shape != (16,):
            raise ValueError(f"Expected state shape (16,), got {state.shape}")
        return state

    def execute_action(self, action: np.ndarray) -> None:
        """
        执行动作。
        输入 action 通常是一个时刻的动作向量 shape=(action_dim,)
        你需要把它映射到真实机械臂控制命令。
        """
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size < 16:
            raise ValueError(f"Expected action dim >= 16, got {action.size}")
        if action.size > 16:
            logging.warning("Action dim=%d > 16, only first 16 dims are used.", action.size)
            action = action[:16]

        left_joint = action[:7]
        left_gripper = action[7]
        right_joint = action[8:15]
        right_gripper = action[15]

        left_code = self._left_arm.move(left_joint.tolist())
        right_code = self._right_arm.move(right_joint.tolist())
        if left_code != 0 or right_code != 0:
            logging.warning(
                "Joint command return code: left=%s, right=%s",
                left_code,
                right_code,
            )

        left_gripper_cmd = int(np.clip(left_gripper, self._gripper_min, self._gripper_max))
        right_gripper_cmd = int(np.clip(right_gripper, self._gripper_min, self._gripper_max))
        self._left_arm.set_gripper_abso(left_gripper_cmd)
        self._right_arm.set_gripper_abso(right_gripper_cmd)

    def move_to_start_pose(self) -> None:
        left_code = self._left_arm.move(START_POSITION_ANGLE_LEFT_ARM)
        right_code = self._right_arm.move(START_POSITION_ANGLE_RIGHT_ARM)
        left_gripper_code = self._left_arm.set_gripper_abso(1000)
        right_gripper_code = self._right_arm.set_gripper_abso(1000)

        if left_code != 0 or right_code != 0 or left_gripper_code != 0 or right_gripper_code != 0:
            raise RuntimeError(
                "Failed to move robot to start pose: "
                f"left_code={left_code}, right_code={right_code}"
                f"left_gripper_code={left_gripper_code}, right_gripper_code={right_gripper_code}"
            )

    def close(self) -> None:
        # SDK 封装中未暴露统一 disconnect，这里释放引用即可。
        self._left_arm = None
        self._right_arm = None


# =========================
# 3. 预处理工具
# =========================
def preprocess_image(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """
    按 openpi 推荐方式：
    - resize_with_pad
    - convert_to_uint8
    """
    return image_tools.convert_to_uint8(
        image_tools.resize_with_pad(img, out_h, out_w)
    )


def build_observation(
    camera: RealCameraInterface,
    robot: RealManStateInterface,
    args: Args,
) -> dict[str, Any]:
    """
    构造发给策略服务端的 observation。
    """

    main_img, left_img, right_img = camera.get_images(
        need_left_wrist=args.use_left_wrist,
        need_right_wrist=args.use_right_wrist,
    )
    obs: dict[str, Any] = {
        "observation/image": preprocess_image(
            main_img, args.image_height, args.image_width
        ),
        "observation/state": robot.get_robot_state(),
        "prompt": args.prompt,
    }

    if args.use_left_wrist:
        if left_img is None:
            raise RuntimeError("left wrist camera is enabled but frame is missing")
        obs["observation/left_wrist_image"] = preprocess_image(
            left_img, args.image_height, args.image_width
        )

    if args.use_right_wrist:
        if right_img is None:
            raise RuntimeError("right wrist camera is enabled but frame is missing")
        obs["observation/right_wrist_image"] = preprocess_image(
            right_img, args.image_height, args.image_width
        )

    return obs


# =========================
# 4. 动作执行策略
# =========================
def execute_action_chunk(
    robot: RealManStateInterface,
    action_chunk: np.ndarray,
    control_hz: float,
    exe_chunk: int=0, 
) -> None:
    """
    执行动作块。
    action_chunk: shape = (action_horizon, action_dim)

    这里给你保留最常见的 open-loop 执行方式：
    逐个 action 发送给机械臂。
    """
    dt = 1.0 / control_hz

    for i, action in enumerate(action_chunk):
        
        logging.info("action %d executing", i+1)
        
        if exe_chunk!=0 and i>=exe_chunk:
            break

        step_t0 = time.perf_counter()

        robot.execute_action(np.asarray(action))

        elapsed = time.perf_counter() - step_t0
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


# =========================
# 5. 主循环
# =========================
def main(args: Args) -> None:
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    logging.info("Initializing real camera interface...")
    camera = RealCameraInterface(args)

    logging.info("Initializing robot state interface...")
    robot = RealManStateInterface(args)
    logging.info("Moving robot to configured start pose before inference...")
    robot.move_to_start_pose()

    logging.info(
        "Connecting to policy server at ws://%s:%d ...",
        args.host,
        args.port,
    )
    client = websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
    )

    try:
        for step in range(args.num_steps):
            loop_t0 = time.perf_counter()

            # 1) 采集 observation
            observation = build_observation(camera, robot, args)

            if args.verbose:
                logging.info(
                    "Step %d | observation keys: %s",
                    step,
                    list(observation.keys()),
                )
                logging.info(
                    "main_image=%s, state=%s",
                    observation["observation/image"].shape,
                    observation["observation/state"].shape,
                )
                if "observation/left_wrist_image" in observation:
                    logging.info(
                        "left_wrist_image=%s",
                        observation["observation/left_wrist_image"].shape,
                    )
                if "observation/right_wrist_image" in observation:
                    logging.info(
                        "right_wrist_image=%s",
                        observation["observation/right_wrist_image"].shape,
                    )

            # 2) 调用策略服务端
            result = client.infer(observation)

            if "actions" not in result:
                raise KeyError(f"Server response does not contain 'actions': {result}")

            action_chunk = np.asarray(result["actions"])
            logging.info("Step %d | action_chunk shape: %s", step, action_chunk.shape)

            # 3) 执行动作块
            execute_action_chunk(robot, action_chunk, args.control_hz,exe_chunk=50)

            loop_ms = (time.perf_counter() - loop_t0) * 1000.0
            logging.info("Step %d finished, total loop time: %.2f ms", step, loop_ms)

    finally:
        logging.info("Closing interfaces...")
        camera.close()
        robot.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
