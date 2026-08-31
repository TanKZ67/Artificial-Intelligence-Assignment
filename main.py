"""
ASL 手语识别 - 主入口

运行方式:
    python main.py

菜单里选择一个模型 (CNN / Random Forest / SVM) 后即可打开摄像头开始识别。
在识别界面里:
    - 按 [ESC]        返回本菜单，重新选择模型
    - 按 [Q]          直接退出整个程序
"""

import sys

from Pipelines.Pipelines import main_cnn
from Pipelines.Pipelines import main_rf
from Pipelines.Pipelines import main_svm

MODELS = {
    "1": ("CNN (MediaPipe + TFLite)", main_cnn.run),
    "2": ("Random Forest", main_rf.run),
    "3": ("SVM", main_svm.run),
}


def print_menu():
    print("\n===== ASL 手语识别 - 选择模型 =====")
    for key, (name, _) in MODELS.items():
        print(f"  [{key}] {name}")
    print("  [Q] exit programe")
    print("====================================")


def main():
    while True:
        print_menu()
        choice = input("Please Choice: ").strip().lower()

        if choice in ("q", "quit", "exit"):
            print("已退出。")
            break

        if choice not in MODELS:
            print("Invalid option, please re-enter.")
            continue

        name, run_pipeline = MODELS[choice]
        print(f"\n启动模型: {name}")
        print("(识别界面中: ESC 返回菜单, Q 直接退出)\n")

        try:
            action = run_pipeline()
        except KeyboardInterrupt:
            print("\n检测到中断，返回菜单。")
            action = "menu"
        except Exception as exc:
            print(f"[错误] 运行 {name} 时出现异常: {exc}")
            action = "menu"

        if action == "quit":
            print("exited。")
            break
        # action == "menu" -> 回到 while 循环，重新显示菜单


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nexited")
        sys.exit(0)
