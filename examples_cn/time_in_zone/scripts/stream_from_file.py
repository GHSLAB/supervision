import os
import subprocess
import tempfile
from glob import glob
from threading import Thread

import yaml
from jsonargparse import auto_cli

SERVER_CONFIG = {"protocols": ["tcp"], "paths": {"all": {"source": "publisher"}}}
BASE_STREAM_URL = "rtsp://localhost:8554/live"


def main(video_directory: str, number_of_streams: int = 6) -> None:
    """
    使用 RTSP 协议推送视频的脚本。

    Args:
        video_directory: 包含待推送视频文件的目录。
        number_of_streams: 要推送的视频文件数量。
    """
    video_files = find_video_files_in_directory(video_directory, number_of_streams)
    try:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_file_path = create_server_config_file(temporary_directory)
            run_rtsp_server(config_path=config_file_path)
            stream_videos(video_files)
    finally:
        stop_rtsp_server()


def find_video_files_in_directory(directory: str, limit: int) -> list:
    """在指定目录中查找视频文件。"""
    video_formats = ["*.mp4", "*.webm"]
    video_paths = []
    for video_format in video_formats:
        video_paths.extend(glob(os.path.join(directory, video_format)))
    return video_paths[:limit]


def create_server_config_file(directory: str) -> str:
    """在给定目录中创建 RTSP 服务器的配置文件。"""
    config_path = os.path.join(directory, "rtsp-simple-server.yml")
    with open(config_path, "w") as config_file:
        yaml.dump(SERVER_CONFIG, config_file)
    return config_path


def run_rtsp_server(config_path: str) -> None:
    """通过 Docker 启动一个 RTSP 服务器。"""
    command = (
        "docker run --rm --name rtsp_server -d -v "
        f"{config_path}:/rtsp-simple-server.yml -p 8554:8554 "
        "aler9/rtsp-simple-server:v1.3.0"
    )
    if run_command(command.split()) != 0:
        raise RuntimeError("无法启动 RTSP 服务器！")


def stop_rtsp_server() -> None:
    """停止正在运行的 RTSP 服务器容器。"""
    run_command("docker kill rtsp_server".split())


def stream_videos(video_files: list) -> None:
    """通过线程将多个视频文件推送到各自的 RTSP URL。"""
    threads = []
    for index, video_file in enumerate(video_files):
        stream_url = f"{BASE_STREAM_URL}{index}.stream"
        print(f"正在将 {video_file} 推送至 {stream_url}")
        thread = stream_video_to_url(video_file, stream_url)
        threads.append(thread)
    for thread in threads:
        thread.join()


def stream_video_to_url(video_path: str, stream_url: str) -> Thread:
    """使用 ffmpeg 将单个视频文件循环推送至指定的 RTSP URL。"""
    command = (
        f"ffmpeg -re -stream_loop -1 -i {video_path} "
        f"-f rtsp -rtsp_transport tcp {stream_url}"
    )
    return run_command_in_thread(command.split())


def run_command_in_thread(command: list) -> Thread:
    """在新线程中执行 shell 命令。"""
    thread = Thread(target=run_command, args=(command,))
    thread.start()
    return thread


def run_command(command: list) -> int:
    """同步执行 shell 命令并返回返回码。"""
    process = subprocess.run(command)  # noqa: S603 # TODO: 验证命令输入以防止执行不可信输入
    return process.returncode


if __name__ == "__main__":
    from jsonargparse import auto_cli, set_parsing_settings

    set_parsing_settings(parse_optionals_as_positionals=True)
    auto_cli(main, as_positional=False)
