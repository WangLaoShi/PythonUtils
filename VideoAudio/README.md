# 视频和音频的相关操作

## moviepy 

身处数据爆炸增长的时代，各种各样的数据都飞速增长，视频数据也不例外。我们可以使用 python 来提取视频中的音频，而这仅仅需要安装一个体量很小的python包，然后执行三行程序！

语音数据在数据分析领域极为重要。比如可以分析语义、口音、根据人的情绪等等。可以应用于偏好分析、谎话检测等等。

### 提取音频 

需要用到 python 包 moviepy，这里是moviepy 的 github 地址

安装 python 包

需要用到 python 包 moviepy，这里是[moviepy 的 github 地址](https://github.com/Zulko/moviepy)

安装 moviepy，cmd 或 bash 输入

`pip install moviepy`


假设有一个 mp4 文件路径为"e:/chrome/my_video.mp4"，我们想提取其音频保存到"“e:/chrome/my_audio.wav”"，那么三行程序为：

```python
from moviepy.editor import AudioFileClip
my_audio_clip = AudioFileClip("e:/chrome/my_video.mp4")
my_audio_clip.write_audiofile("e:/chrome/my_audio.wav")

```

### 分析音频

可以使用 librosa 包来分析音频，这里是[librosa 的 github 地址](https://github.com/librosa/librosa)

#### 安装 python 包

安装 librosa，cmd 或 bash 输入

```prism
pip install librosa
```

**需要说明**，librosa 包本身不支持 MP3 格式，需要一些相关包的支持。官网上说使用 conda 安装则自动安装 MP3 支持的相关包。具体请去[librosa 的 github 地址](https://github.com/librosa/librosa)了解。

#### 读取音频

假设有一个 wav 文件路径为"e:/chrome/my_audio.wav"。科普一下音频数据的内容，可以认为记录**采样频率**和**每个采样点的信号强度**两个部分即可构成一个音频文件。数据流可理解为一个数组，按照字节存储。  
下面我们读取音频

```prism
import librosa
audio, freq = librosa.load('e:/chrome/my_audio.wav')
time = np.arange(0, len(audio)) / freq
print(len(audio), type(audio), freq, sep="\t")
```


下图是我电脑的示例，可以看到读取到了**采样频率**和**每个采样点的信号强度**，采样点共 2121210，频率为 22050，音频长度约 96 秒

#### matplotlib 画信号强度图

```python

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(time, audio)
ax.set(xlabel='Time(s)', ylabel='Sound Amplitude')
plt.show()
```

![ESNllB](https://upiclw.oss-cn-beijing.aliyuncs.com/uPic/ESNllB.jpg)

#### librosa 画信号强度图

当然我们可以使用 librosa 库的工具来分析，可以修掉音频首尾的其他信息，画信号强度图的方式如下：

```python
import  librosa.display
audio, _ = librosa.effects.trim(audio)#Trim leading and trailing #silence from an audio signal.
librosa.display.waveplot(audio, sr=freq)
plt.show()
```

![RjsiZ4](https://upiclw.oss-cn-beijing.aliyuncs.com/uPic/RjsiZ4.jpg)

## pydub

[项目地址](https://github.com/jiaaro/pydub)

## 提取字幕的几个方法和算法

[基于图像识别和文字识别用 Python 提取视频字幕](https://xinancsd.github.io/Python/subtitle.html)





