class OrbbecCamera(BaseCamera):
    """Orbbec摄像头设备实现"""
    def __init__(self, camera_type: str, camera_serial: str, camera_position:str):
        super().__init__(camera_type, camera_serial, camera_position)
        self.pipeline = None
        self.config = None

    def connect(self, **kwargs) -> bool:
        """连接RealSense摄像头"""
        try:
            self.pipeline = Pipeline()
            self.config.enable_device(CAMERA_SERIALS[self.camera_type][self.device_position])
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.pipeline.start(self.config)
            self.logger_msg(f"connected successfully")
            return True
        except Exception as e:
            self.logger_msg(f"connect failed: {str(e)}")
            return False