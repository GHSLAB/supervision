import os

from supervision.assets import VideoAssets, download_assets

# 若 data 目录不存在则创建，然后切换到该目录并下载示例视频。
if not os.path.exists("data"):
    os.makedirs("data")
os.chdir("data")
download_assets(VideoAssets.VEHICLES)
