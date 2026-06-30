#!/bin/bas

# 设置显示输出
export DISPLAY=:0

# 等待 X 服务器就绪（最多等待 15 秒）
for i in {1..15}; do
    if xset q &>/dev/null; then
        break
    fi
    sleep 1
done

# 进入程序目录
cd /home/abysm/camera

# ===== GPIO 初始化 =====
# 使用 sudo 执行初始化脚本（无需密码，已通过 sudoers 放行）
sudo /home/abysm/camera/setup_gpio.sh

# 运行主程序
exec /home/abysm/demo.env/bin/python3 /home/abysm/camera/zj9.py
