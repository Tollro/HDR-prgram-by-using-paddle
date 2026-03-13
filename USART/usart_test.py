from periphery import Serial
import time

# 初始化串口
uart = Serial("/dev/ttyS3", baudrate=9600, databits=8, parity="none", stopbits=1)

def send_at_command(command, expected_response="OK", timeout=2):
    """发送 AT 指令并等待响应"""
    uart.write((command + "\r\n").encode('utf-8'))
    start_time = time.time()
    response = ""
    while time.time() - start_time < timeout:
        data = uart.read(1024, timeout=0.1)
        if data:
            response += data.decode('utf-8', errors='ignore')
            if expected_response in response:
                return True, response
    return False, response

try:
    print("初始化 HC-05 主模式连接...")

    # 查询当前角色（确认是否为主模式）
    success, response = send_at_command("AT+ROLE?", "ROLE=1")
    if not success:
        print(f"查询角色失败: {response}")
        exit()

    # 设置目标设备的蓝牙地址（请替换为目标设备的实际地址）
    target_address = "98D3:41:F6FB76"  # 替换为目标设备的蓝牙地址
    print(f"尝试连接到目标设备: {target_address}")

    # 发送连接指令
    success, response = send_at_command(f"AT+PAIR={target_address},10", "PAIRED")
    if not success:
        print(f"配对失败: {response}")
        exit()

    success, response = send_at_command(f"AT+LINK={target_address}", "CONNECTED")
    if not success:
        print(f"连接失败: {response}")
        exit()

    print("连接成功！开始数据通信...")

    # 数据通信循环
    while True:
        # 发送数据到目标设备
        message = input("请输入要发送的消息: ")
        uart.write(message.encode('utf-8'))
        
        # 简单延时避免过快循环
        time.sleep(0.1)

        # 接收来自目标设备的数据
        data = uart.read(1024, timeout=1)
        if data:
            print(f"接收到的数据: {data.decode('utf-8', errors='ignore')}")

except Exception as e:
    print(f"发生错误: {e}")

finally:
    uart.close()
    print("串口已关闭")
