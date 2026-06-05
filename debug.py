import sys
# 强行锁定你正在修改的这个源码目录
sys.path.insert(0, r"E:\projects\论文\yolov8\yolov8\code\yolo\ultralytics")

print("--- 诊断开始 ---")
try:
    print("正在测试 1: 导入底层 block 模块...")
    from ultralytics.nn import modules
    print("【成功】底层 modules/block 导入正常！")
except Exception as e:
    print("【崩溃】问题出在 nn/modules 的修改上！详细错误如下：")
    import traceback
    traceback.print_exc()

print("\n---------------------------------")
try:
    print("正在测试 2: 导入解析器所在的 tasks 模块...")
    from ultralytics.nn import tasks
    print("【成功】tasks.py 导入正常！")
except Exception as e:
    print("【崩溃】问题出在 tasks.py 的修改上！详细错误如下：")
    import traceback
    traceback.print_exc()