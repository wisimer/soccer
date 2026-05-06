import av

path = '../data/HBTFZwMdcCw.mp4'
# 打开音频/视频文件
container = av.open(path)

video_stream = container.streams.video[0]  # 对于视频文件
audio_stream = container.streams.audio[0]  # 对于音频文件


audio_frames = []
video_frames = []
# 解码并处理视频帧
index = 0
for frame in container.decode(video_stream):
    # 读取视频帧
    print(f'帧索引，帧宽度：{frame.width}，帧高度：{frame.height}，帧格式：{frame.format}，时间戳: {round(frame.time,2)}，Frame Time Base: {frame.time_base}')
    video_frames.append(frame)
    index += 1
    if index == 15:
        print(f'开始保存第16帧：')
        frame.to_image().save('output.jpg')

for frame in container.decode(audio_stream):
    # 读取音频帧（通道布局：表示声音的通道配置，如立体声（stereo），单声道（mono），5.1声道等。）
    print(f'帧索引，帧格式：{frame.format}，采样率：{frame.rate}，通道格式：{frame.layout}，时间戳: {frame.pts}，时间基准: {frame.time_base}，音频样本数目: {frame.samples}')
    audio_frames.append(frame)


# 编码帧（将视频帧和音频帧对应组合）
for audio_frame, video_frame in zip(audio_frames, video_frames):
    container.mux(audio_frame.encode(), video_frame.encode())


# 关闭容器
container.close()
