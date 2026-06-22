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

START_POSITION_ANGLE_RIGHT_ARM = [
    18.363,
    14.513,
    -6.523,
    100.689,
    2.139,
    63.749,
    90.0,
]



# =========================
# 1. 参数定义
# =========================
@dataclasses.dataclass
class Args:
    host: str = "192.168.0.189"
    port: int = 11257
    api_key: str | None = None

    num_steps: int = 200
    control_hz: float = 10.0
    exe_chunk: int = 20 #50

    # 图像输入尺寸，通常 pi0/pi0.5 预训练常用 224
    image_height: int = 224
    image_width: int = 224

    # 任务文本
    prompt: str = "insert usb into red hole"

    # 是否启用双腕相机
    use_left_wrist: bool = False
    use_right_wrist: bool = True

    # 调试
    verbose: bool = True

    # RealSense 序列号（默认复用 CameraManager 里的配置）
    camera_type: str = "RealSense"
    main_camera_serial: str = CAMERA_SERIALS["RealSense"]["head"]
    right_wrist_camera_serial: str = CAMERA_SERIALS["RealSense"]["right_wrist"]

    # 右臂 IP
    arm_ip: str = "192.168.0.17"

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

        self._main_key = f"{args.camera_type}_head"
        self._right_key = f"{args.camera_type}_right_wrist"

        camera_configs = [
            {
                "type": args.camera_type,
                "position": "head",
                "serial": args.main_camera_serial,
            },
            {
                "type": args.camera_type,
                "position": "right_wrist",
                "serial": args.right_wrist_camera_serial,
            },
        ]

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

    def get_images(self) -> tuple[np.ndarray, np.ndarray]:
        frames = self._manager.get_frames()
        main_img = self._extract_rgb(frames, self._main_key, "main camera")
        right_img = self._extract_rgb(frames, self._right_key, "right wrist camera")
        return main_img, right_img

    def close(self) -> None:
        for camera in self._manager.get_cameras().values():
            try:
                camera.disconnect()
            except Exception as exc:  # noqa: BLE001
                logging.warning("Failed to disconnect camera: %s", exc)


class RealManStateInterface:
    """右臂状态与动作接口。"""

    def __init__(self, args: Args) -> None:
        self._arm = DeviceRealManController(args.arm_ip)
        self._gripper_min = float(args.gripper_min)
        self._gripper_max = float(args.gripper_max)

    def get_robot_state(self) -> np.ndarray:
        """返回单臂 state: eef(6) + gripper => 7 维。"""
        eef, joint, gripper, time = self._arm.get_state()

        eef = np.asarray(eef, dtype=np.float32).reshape(-1)
        state = np.concatenate(
            [eef, np.asarray([gripper], dtype=np.float32)],
            axis=0,
        )
        if state.shape != (7,):
            raise ValueError(f"Expected single-arm state shape (7,), got {state.shape}")
        return state

    def execute_action(self, action: np.ndarray) -> None:
        """执行右臂 7 维动作"""
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size < 7:
            raise ValueError(f"Expected action dim >= 7, got {action.size}")
        if action.size > 7:
            logging.warning("Action dim=%d > 7, only first 7 dims are used.", action.size)
            action = action[:7]

        print("action:", action)

        exe_eef = action[:6]
        exe_gripper = action[6]

        code = self._arm.arm_controller.rm_movej_p(exe_eef.tolist(), 20, 0, 0, 0)
        if code != 0:
            logging.warning("Right arm command return code: %s", code)

        exe_gripper = int(np.clip(exe_gripper, self._gripper_min, self._gripper_max))
        gripper_code = self._arm.set_gripper_abso(exe_gripper)
        if gripper_code != 0:
            logging.warning("Right gripper command return code: %s", gripper_code)

    def move_to_start_pose(self) -> None:
        code = self._arm.move(START_POSITION_ANGLE_RIGHT_ARM)
        gripper_code = self._arm.set_gripper_abso(150)

        if code != 0 or gripper_code != 0:
            raise RuntimeError(
                "Failed to move robot to start pose: "
                f"move_code={code}, gripper_code={gripper_code}"
            )

    def close(self) -> None:
        self._arm = None


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
    head_img, wrist_img = camera.get_images()
    head_img = preprocess_image(head_img, args.image_height, args.image_width)
    wrist_img = preprocess_image(wrist_img, args.image_height, args.image_width)
    zero_img = np.zeros_like(head_img, dtype=np.uint8)

    return {
        "observation/image": head_img,
        "observation/wrist_image": wrist_img,
        "observation/state": robot.get_robot_state(),
        "prompt": args.prompt,
    }


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
        if exe_chunk != 0 and i >= exe_chunk:
            break
        
        if (i+1) % 3 == 0:
            continue

        logging.info("action %d executing", i + 1)
        step_t0 = time.perf_counter()
        robot.execute_action(np.asarray(action))

        sleep_time = dt - (time.perf_counter() - step_t0)
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

    logging.info("Connecting to policy server at ws://%s:%d ...", args.host, args.port)
    client = websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
    )

    try:
        for step in range(args.num_steps):
            loop_t0 = time.perf_counter()
            observation = build_observation(camera, robot, args)

            if args.verbose:
                logging.info("Step %d | observation keys: %s", step, list(observation.keys()))
                logging.info(
                    "image=%s, wrist_image=%s, state=%s",
                    observation["observation/image"].shape,
                    observation["observation/wrist_image"].shape,
                    observation["observation/state"].shape,
                )

            result = client.infer(observation)
            if "actions" not in result:
                raise KeyError(f"Server response does not contain 'actions': {result}")

            action_chunk = np.asarray(result["actions"], dtype=np.float32)
            logging.info("Step %d | action_chunk shape: %s", step, action_chunk.shape)
            execute_action_chunk(robot, action_chunk, args.control_hz, exe_chunk=args.exe_chunk)

            loop_ms = (time.perf_counter() - loop_t0) * 1000.0
            logging.info("Step %d finished, total loop time: %.2f ms", step, loop_ms)

    finally:
        logging.info("Closing interfaces...")
        camera.close()
        robot.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
