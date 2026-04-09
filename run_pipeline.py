import os
import sys

def run_command(cmd):
    print(f"Running: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        print(f"Command failed with exit code {ret}")
        sys.exit(ret)

def main():
    print("=== Starting Data Pipeline ===")
    run_command("python src/data_generation.py")
    run_command("python src/data_cleaning.py")
    run_command("python src/eda_analysis.py")
    run_command("python src/hotspot_detection.py")
    run_command("python src/model_training.py")
    print("=== Pipeline Complete ===")

if __name__ == "__main__":
    main()
