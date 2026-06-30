#!/bin/bash
for pin in 70 71 72 73 91 92; do
    gpio_path="/sys/class/gpio/gpio${pin}"
    if [ ! -d "$gpio_path" ]; then
        # 导出引脚并设置为输入
        echo $pin > /sys/class/gpio/export
        echo "in" > /sys/class/gpio/gpio${pin}/direction
    fi
    # 修复真实路径权限（每次开机都做，确保可用）
    real_dir=$(readlink -f "$gpio_path" 2>/dev/null)
    if [ -n "$real_dir" ]; then
        chown -R root:video "$real_dir"
        chmod -R u+rwX,g+rwX,o-rwx "$real_dir"
    fi
done
echo "✅ GPIO 已初始化，普通用户可读写"
