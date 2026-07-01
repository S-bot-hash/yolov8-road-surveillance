import subprocess
import time

# 按照你希望执行的顺序，把脚本名称放进列表里
experiments = [
    "baseline.py",
    "exp-1-GhostC2f.py",
    "exp-2-NeckOnly-GhostC2f.py",
    "exp-3-GhostC2f-SEAtt.py",
    "exp-4-NeckOnly-GhostC2f-SEAtt.py",
]


def run_experiments():
    print(f"🚀 准备开始自动化执行 {len(experiments)} 个消融实验...\n" + "=" * 50)

    success_list = []
    fail_list = []

    for i, exp_script in enumerate(experiments):
        print(f"\n[{i + 1}/{len(experiments)}] 正在启动实验: {exp_script} ...")
        start_time = time.time()

        try:
            # 使用 subprocess 调用 python 运行目标脚本
            # check=True 表示如果脚本报错，会抛出 CalledProcessError 异常
            subprocess.run(["python", exp_script], check=True)

            # 计算耗时
            elapsed_time = (time.time() - start_time) / 3600
            print(f"✅ 实验 {exp_script} 成功完成！耗时: {elapsed_time:.2f} 小时。")
            success_list.append(exp_script)

        except subprocess.CalledProcessError:
            print(f"❌ 实验 {exp_script} 运行失败！跳过并准备执行下一个。")
            fail_list.append(exp_script)
        except FileNotFoundError:
            print(f"⚠️ 找不到文件 {exp_script}，请检查文件名是否正确！")
            fail_list.append(exp_script)

    # 最终输出总结报告
    print("\n" + "=" * 50)
    print("🎉 所有列出的消融实验已全部执行完毕！")
    print(f"✅ 成功完成 ({len(success_list)}): {', '.join(success_list) if success_list else '无'}")
    print(f"❌ 失败或跳过 ({len(fail_list)}): {', '.join(fail_list) if fail_list else '无'}")
    print("=" * 50)


if __name__ == "__main__":
    run_experiments()
