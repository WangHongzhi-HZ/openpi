import threading
import time
import numpy as np

from Robotic_Arm.rm_robot_interface import *

class RealManController:
    def __init__(self, rm_ip, thread_mode=rm_thread_mode_e.RM_TRIPLE_MODE_E):
        # import pdb
        # pdb.set_trace()
        try:
            result = self.initialize_robot(rm_ip, thread_mode)
            if result is None:
                raise ConnectionError(f"Failed to connect to robot arm at {rm_ip}")
            self.arm_controller, self.handle = result
        except Exception as e:
            raise ConnectionError(f"Failed to initialize robot arm: {str(e)}")
        
        # 机械臂IP
        self.rm_ip = rm_ip
        
        # 机械臂状态
        self.eef = [0,0,0,0,0,0]        # 末端位置
        self.joint = [0,0,0,0,0,0,0]    # 关节位置
        self.gripper = 0                # 夹爪状态
        self.update_time = None         # 状态更新时间

        # 机械臂锁
        self.lock = threading.Lock()

        # =================== 遥操控制 ====================
        # --------------- 遥操作原理 -----------------------
        # ***  机械臂实际位置 = 机械臂初始位置 + 末端相对位置  **
        # ***  末端相对位置  = 手柄实际位置 - 手柄初始位置    **
        # ------------------------------------------------
        self.is_controlling = False
        self.prev_tech_state = None     # 手柄初始位置
        self.arm_first_state = None     # 机械臂初始位置
        self.delta = [0,0,0,0,0,0]      # 机械臂相对初始位置的变化 = tech_state - prev_state
        self.gripper_close = False
        
        self.q_last = None
    

    # 初始化机械臂
    def initialize_robot(self,ip, mode=None):
        robot = RoboticArm(mode)
        handle = robot.rm_create_robot_arm(ip, 8080) 
        return  robot, handle

    # 更新机械臂状态
    def update_state(self):
        # with self.lock:
        state = self.arm_controller.rm_get_current_arm_state()
        gripper = self.arm_controller.rm_get_gripper_state()

        # print("state:", state)
        # print("gripper:", gripper)

        # print("gripper:", gripper)
        self.gripper = gripper[1]["actpos"]
        self.eef = state[1]["pose"]
        self.joint = state[1]["joint"]
        if state[0]!=0 or gripper[0]!=0:
            print(f"[ERROR] 获取机械臂状态失败,state:{state}, gripper:{ gripper}")

        self.update_time = time.time()
        # print(f"eef:{self.eef},type:{type(self.eef)},type_per:{type(self.eef[0])}")

    # 获取当前状态
    def get_state(self):
        self.update_state()
        return self.eef,self.joint,self.gripper,self.update_time
    
    # 设置夹爪位置——绝对位置
    def set_gripper_abso(self, pos):
        """设置夹爪位置,pos范围0-1000"""
        # with self.lock:
        #     res = self.arm_controller.rm_set_gripper_pos(pos)
        res = self.arm_controller.rm_set_gripper_position(int(pos),True,10)
        if res!=0:
            print("[ERROR] 设置夹爪位置失败,错误码:", res)
        
        return res
    # 设置夹爪位置——相对位置
    def set_gripper_rela(self, gripper_vari):
        with self.lock:
            new_pos = self.arm_controller.rm_get_gripper_state()[1]["actpos"] + gripper_vari*1000
        if new_pos > 1000:
            new_pos = 1000
        # print(f"gripper_vari:{gripper_vari},new_pos: {new_pos}")
        # import pdb
        # pdb.set_trace()
        # with self.lock:
        res = self.arm_controller.rm_set_gripper_position(int(new_pos),True,10)
        print(f"res: {res}")
    
    # 设置夹爪——quest遥操 0/1夹爪
    def set_gripper_quest(self, quest_gripper:float):
        if quest_gripper < 0.20 and not self.gripper_close:
            with self.lock:
                success = self.arm_controller.rm_set_gripper_pick(500, 1000, True, 0)
            self.gripper_close = True
        elif quest_gripper > 0.8 and self.gripper_close:
            with self.lock:
                success = self.arm_controller.rm_set_gripper_release(500, True, 0)        
            self.gripper_close = False

    
    # 透传/增量 末端移动
    def move_quest(self, tech_state):
        for i in range(6):
            self.delta[i] = tech_state[i]-self.prev_tech_state[i]
        
        next_state = [self.arm_first_state[i] + self.delta[i] for i in range(6)] 

        
        # with self.lock:
        success = self.arm_controller.rm_movep_canfd(next_state,False,0,80)
        # success = self.arm_controller.rm_movej(next_state,20,0,0,1)
        # success = self.arm_controller.rm_movep_canfd(next_state,True,0,80)
        # success = self.arm_controller.rm_movep_follow(next_state)
        # print(f"first_state:{self.arm_first_state}")
        # print(f"delta:{self.delta}")
        # print(f"next_state:{next_state},Success: {success}")
    
    # 机械臂控制
    def move(self, joint):
        with self.lock:
            success = self.arm_controller.rm_movej(joint,20,0,0,1)
        return success

def initialize_realman():
    """
    初始化 RealMan 机械臂到预设位置
    """
    global left_realman_controller, right_realman_controller
    
    START_POSITION_ANGLE_LEFT_ARM = [
        88, -42, -7, -83, -45, -79, 80
    ]
    
    START_POSITION_ANGLE_RIGHT_ARM = [
        -74, 41, -4, 84, 46, 74, 103
    ]

    # 设置左臂初始位置
    left_signal = left_realman_controller.move(START_POSITION_ANGLE_LEFT_ARM)
    debug_print(f"左臂初始位置设置为: {START_POSITION_ANGLE_LEFT_ARM}", True)

    # 设置右臂初始位置
    right_signal = right_realman_controller.move(START_POSITION_ANGLE_RIGHT_ARM)
    debug_print(f"右臂初始位置设置为: {START_POSITION_ANGLE_RIGHT_ARM}", True)

    if left_signal != 0 and right_signal != 0:
        debug_print(f"机械臂初始位置设置失败---rightarm:{right_signal}---leftarm:{left_signal}，请手动示教机械臂", True)
        return False

    # 等待机械臂到达初始位置
    time.sleep(2)
    debug_print("机械臂初始位置设置成功", True)
    return True



if __name__ == "__main__":
    rm_ip = "192.168.0.18"
    rm_controller = RealManController(rm_ip)
    print(f"机械臂状态：{rm_controller.get_state()}")


