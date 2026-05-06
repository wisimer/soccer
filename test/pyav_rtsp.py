import av

# RTSP 流的地址（请替换为你自己的）
url = "rtsp://localhost:8554/soccer"

# 可以添加特定选项，如指定 TCP 传输
options = {'rtsp_transport': 'tcp'}

container = av.open(url, 'r', options=options)

for frame in container.decode(video=0):
    # 处理和显示帧
    print(f'帧索引，{frame.pts},  帧宽度：{frame.width}，帧高度：{frame.height}，帧格式：{frame.format}，时间戳: {round(frame.time,2)}，Frame Time Base: {frame.time_base}')


container.close()