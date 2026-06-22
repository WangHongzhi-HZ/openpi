import threading
import time
import numpy as np

from Robotic_Arm.rm_robot_interface import *

from .Realman_IK.ik_rbtdef import *
from .Realman_IK.ik_rbtutils import *
from .Realman_IK.ik_qp import *

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
        self.set_ik(rm_ip)


    def set_ik(self, rm_ip):
        dT = 0.02
        if(rm_ip == "192.168.0.18"):
            self.ik = QPIK("RM75B", dT)
            self.ik.set_install_angle([45, 0, 180], 'deg')
            self.ik.set_tool_cs_params([0, 0, 0.1, -1.57, -1.57, 0])
            self.ik.set_7dof_elbow_max_angle(-3, 'deg')
            self.ik.set_7dof_q3_min_angle(30, 'deg')
            self.ik.set_7dof_q3_max_angle(-30, 'deg')
            self.ik.set_dq_max_weight([1.0, 1.0, 1.0, 1.0, 0.1, 1.0, 1.0])
        elif(rm_ip == "192.168.0.19"):
            self.ik = QPIK("RM75B", dT)
            self.ik.set_install_angle([45, 0, 0], 'deg')
            self.ik.set_tool_cs_params([0, 0, 0.1, 0.524, -1.57, 1.57])
            self.ik.set_7dof_elbow_min_angle(3, 'deg')
            self.ik.set_7dof_q3_min_angle(-30, 'deg')
            self.ik.set_7dof_q3_max_angle(30, 'deg')
            self.ik.set_dq_max_weight([1.0, 1.0, 1.0, 1.0, 0.1, 1.0, 1.0])  
    # 初始化机械臂
    def initialize_robot(self,ip, mode=None):
        robot = RoboticArm(mode)
        handle = robot.rm_create_robot_arm(ip, 8080) 
        return  robot, handle

    # 更新机械臂状态
    def update_state(self):
        with self.lock:
            state = self.arm_controller.rm_get_current_arm_state()
            gripper = self.arm_controller.rm_get_gripper_state()

        # state = self.arm_controller.rm_get_current_arm_state()
        # gripper = self.arm_controller.rm_get_gripper_state()

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
        res = self.arm_controller.rm_set_gripper_pos(pos)
        if res!=0:
            print("[ERROR] 设置夹爪位置失败,错误码:", res)
    # 设置夹爪位置——相对位置
    def set_gripper_rela(self, gripper_vari):
        with self.lock:
            new_pos = self.arm_controller.rm_get_gripper_state()[1]["actpos"] + gripper_vari*1000
        if new_pos > 1000:
            new_pos = 1000
        print(f"gripper_vari:{gripper_vari},new_pos: {new_pos}")
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

    # 透传/增量 末端移动
    def move_quest_ik(self, tech_state):
        for i in range(6):
            self.delta[i] = tech_state[i]-self.prev_tech_state[i]
        
        next_state = [self.arm_first_state[i] + self.delta[i] for i in range(6)] 

        Td = pose_to_matrix(next_state[0], next_state[1], next_state[2], next_state[3], next_state[4], next_state[5])
        # print(f"Td: {Td}")
        if self.q_last is None:
        # 第一次：从真实机械臂获取当前关节角
            code, joint_deg = self.arm_controller.rm_get_joint_degree()
            if code != 0:
                raise RuntimeError("获取机械臂关节角失败")
            self.q_last = np.deg2rad(joint_deg)
            
            # q_solve = self.ik.sovler(self.q_last,Td)
            # self.q_last = q_solve
            # print("8[DEBUG] 初始化 IK 解 q_solve (deg):", np.degrees(q_solve))
            return
        q_solve = self.ik.sovler(self.q_last,Td)

        # 计算 FK，检查误差
        T_check = self.ik.fkine(q_solve)
        pos_err = np.linalg.norm(Td[:3, 3] - T_check[:3, 3]) * 1000  # mm
        cos_angle = (np.trace(Td[:3, :3].T @ T_check[:3, :3]) - 1) / 2
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 避免数值误差
        rot_err = np.degrees(np.arccos(cos_angle))

        self.q_last = q_solve
        self.arm_controller.rm_movej_canfd(np.degrees(q_solve).tolist(),False,0,0,50)
        
        
        # with self.lock:
        #     success = self.arm_controller.rm_movep_canfd(next_state,False,0,80)
        # success = self.arm_controller.rm_movej(next_state,20,0,0,1)
        # success = self.arm_controller.rm_movep_canfd(next_state,True,0,80)
        # success = self.arm_controller.rm_movep_follow(next_state)
        # print(f"first_state:{self.arm_first_state}")
        # print(f"delta:{self.delta}")
        # print(f"next_state:{next_state},Success: {success}")  
    
    # 机械臂控制
    def move(self, joint,v=20,block=1):
        with self.lock:
            success = self.arm_controller.rm_movej(joint,v,0,0,1)
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

# -----IK--------
def pose_to_matrix(x, y, z, roll, pitch, yaw):
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0,            0,           1]
    ])
    Ry = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [ 0,             1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rx = np.array([
        [1, 0,            0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])
    R = Rz @ Ry @ Rx  # 旋转矩阵

    # 齐次变换矩阵
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


if __name__ == "__main__":
    rm_ip = "192.168.0.18"
    rm_controller = RealManController(rm_ip)
    print(f"机械臂状态：{rm_controller.get_state()}")


