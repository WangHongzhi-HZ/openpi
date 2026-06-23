"""Quick test to check if the RealMan gripper responds at all."""
import sys
sys.path.insert(0, "/home/ym-gpu-3/data/hongzhi/openpi")

from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

IP = "192.168.0.17"  # right arm IP from client

robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
handle = robot.rm_create_robot_arm(IP, 8080)
print(f"Connected, handle={handle}")

# Test 1: Read gripper state
code, state = robot.rm_get_gripper_state()
print(f"Gripper state: code={code}, state={state}")

# Test 2: Non-blocking set position
for pos in [500, 200, 800, 300]:
    code = robot.rm_set_gripper_position(pos, False, 0)  # non-blocking, immediate return
    print(f"rm_set_gripper_position(pos={pos}, block=False, timeout=0) -> {code}")

# Test 3: Blocking set position
code = robot.rm_set_gripper_position(500, True, 5)
print(f"rm_set_gripper_position(pos=500, block=True, timeout=5) -> {code} (expect timeout=-4)")

print("Done")
