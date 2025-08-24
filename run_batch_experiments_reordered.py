#!/usr/bin/env python
"""
배치 사이즈(256, 64, 1024)에 따른 실험을 차례대로 실행하는 스크립트
"""
import os
import subprocess
import argparse
from pathlib import Path

def run_experiment(config_path, python_path=None):
    """
    지정된 설정 파일을 사용하여 main.py를 실행합니다.
    
    Args:
        config_path: 설정 파일 경로
        python_path: Python 실행 파일 경로 (없으면 'python' 사용)
    """
    print(f"{'='*50}")
    print(f"실험 시작: {config_path}")
    print(f"{'='*50}")
    
    # 프로젝트 루트 디렉토리와 src 디렉토리 경로 설정
    project_root = Path("/workspace/Experiment 1-2025/exp_1")
    src_dir = project_root / "src"
    
    # 명령어 구성
    python_cmd = python_path if python_path else "python"
    
    # 현재 디렉토리를 src로 변경
    cwd = os.getcwd()
    os.chdir(src_dir)
    
    # 환경 변수 설정 - 경고 메시지 방지
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    
    try:
        # 서브프로세스 실행 - shell=False로 설정하고 리스트 형태로 명령어 전달
        process = subprocess.run(
            [python_cmd, "-u", "main.py", "--config", str(config_path)], 
            shell=False, 
            check=True, 
            text=True,
            env=env
        )
        print(f"실험 완료: {config_path} (종료 코드: {process.returncode})")
    except subprocess.CalledProcessError as e:
        print(f"실험 실패: {config_path} (오류 코드: {e.returncode})")
    except KeyboardInterrupt:
        print(f"실험 중단: {config_path}")
    finally:
        # 원래 디렉토리로 복원
        os.chdir(cwd)
    
    print(f"{'='*50}\n")

def main():
    parser = argparse.ArgumentParser(description="배치 사이즈 실험 실행 스크립트")
    parser.add_argument("--python", type=str, default="/workspace/venv39/bin/python",
                        help="Python 실행 파일 경로 (기본값: /workspace/venv39/bin/python)")
    args = parser.parse_args()
    
    # 프로젝트 루트 디렉토리와 configs 디렉토리 경로 설정
    project_root = Path("/workspace/Experiment 1-2025/exp_1")
    configs_dir = project_root / "configs"
    
    # 256 -> 64 -> 1024 순서로 배치 사이즈 설정
    batch_sizes = [256, 64, 1024]
    
    print(f"배치 사이즈 실험을 시작합니다 (256 -> 64 -> 1024 순): {batch_sizes}")
    
    # 각 배치 사이즈에 대해 실험 실행
    for batch_size in batch_sizes:
        config_path = configs_dir / f"configs_batch_{batch_size}.yaml"
        # 설정 파일이 존재하는지 확인
        if not config_path.exists():
            print(f"경고: 설정 파일을 찾을 수 없습니다: {config_path}")
            continue
        
        # 실험 실행
        run_experiment(config_path, args.python)
    
    print("모든 실험이 완료되었습니다.")

if __name__ == "__main__":
    main()
