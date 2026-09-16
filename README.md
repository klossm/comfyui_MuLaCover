comfyui_MuLaCover — ComfyUI 自定义节点

MuLaCover（HeartMuLa 翻唱/重混音生成模型）的 ComfyUI 封装。
官方仓库：https://github.com/HeartMuLa/MuLaCover

输入一段参考音频（或旋律/和弦 MIDI）+ 新歌词 + 风格标签，生成保留原曲结构、
按新歌词演唱的翻唱歌曲。本仓库已包含官方推理包源码（MuLaCover/ 目录），
clone 本插件即可使用，无需单独获取。


【一、功能特性】

1. 参考音频模式：参考曲 → 自动转谱（旋律/和弦/鼓点/BPM）→ 翻唱生成
2. MIDI 模式：直接喂 melody/chord MIDI → 跳过转谱模型，加载更快、结构完全可控
3. CPU 内存常驻：各模型在阶段间隙转存系统内存，二次生成免磁盘重载
   （官方默认每次推理完丢弃模型、下次从磁盘重新加载）
4. 单行实时日志：控制台原地刷新推理步数/速度/剩余时间，不刷屏
5. 节点进度条：内部推理进度同步到 ComfyUI 绿色进度条
6. MIDI 上传按钮：加载 MIDI 节点内置下拉选择 + 本地文件上传（机制同官方 LoadAudio）


【二、节点一览】

MuLaCover 模型加载 —— 加载生成管线；keep_in_ram 控制 CPU 内存常驻
MuLaCover 风格标签 —— 四字段拼装风格串：topic / genre / instrument / mood
MuLaCover 加载 MIDI —— 下拉选择 / 上传本地 .mid 文件
MuLaCover 生成 —— 主节点：歌词+标签+参考音频或 MIDI → 音频；参考音频模式可导出转谱 MIDI


【三、前置要求】

· ComfyUI（新版前后端均可）
· Git
· 显存约 12GB；开启 keep_in_ram 时建议系统内存 ≥ 32GB
· 参考音频模式额外需要转谱模型权重（约 10GB 磁盘，见模型下载一节）


【四、安装】

1. 克隆插件

    cd ComfyUI/custom_nodes
    git clone https://github.com/klossm/comfyui_MuLaCover.git
    cd comfyui_MuLaCover

2. 安装依赖（必须装进 ComfyUI 正在使用的 Python 环境，不是系统 Python）

便携版 / 嵌入式（Windows，在插件目录内执行）：

    ..\..\python_embeded\python.exe -m pip install -e "./MuLaCover[audio]"
    ..\..\python_embeded\python.exe -m pip install mido

venv / 系统环境用户：激活对应环境后执行同样的两条 pip 命令。
若启动报缺依赖，对照 MuLaCover 官方 README 的安装段落补装即可。

3. 下载模型（见下一节）

4. 重启 ComfyUI，浏览器 Ctrl+F5 强刷一次（前端脚本有缓存）


【五、模型下载】

所有权重放到 ComfyUI/models/MuLaCover/
（Linux 需小写 mulacover，或设环境变量 MULACOVER_MODEL_ROOT 指向自定义位置）

目录结构：

    ComfyUI/models/MuLaCover/
    ├── MuLaCover/                            ← 主生成模型 HeartMuLa-3B
    │   ├── config.json
    │   ├── gen_config.json
    │   ├── tokenizer.json
    │   ├── model.safetensors.index.json
    │   └── model-0000X-of-00005.safetensors  ← 共 5 个分片
    ├── HeartCodec-oss/                       ← 音频编解码器
    ├── Qwen3-Embedding-0.6B/                 ← 风格标签编码器
    └── SymbolicTranscriptor/                 ← 音频转谱（仅参考音频模式需要）
        ├── yourmt3/
        │   └── last.ckpt
        └── chord/
            ├── joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s0.best.sdict
            ├── joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s1.best.sdict
            ├── joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s2.best.sdict
            ├── joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s3.best.sdict
            └── joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s4.best.sdict

下载命令（在 ComfyUI/ 根目录执行）。
没有 hf CLI 就先安装：pip install -U "huggingface_hub[cli]"
国内网络可加环境变量 HF_ENDPOINT=https://hf-mirror.com 加速

（1）主生成模型
    https://huggingface.co/HeartMuLa/MuLaCover

    hf download HeartMuLa/MuLaCover --local-dir ComfyUI/models/MuLaCover/MuLaCover

（2）音频编解码器
    https://huggingface.co/HeartMuLa/HeartCodec-oss-20260123

    hf download HeartMuLa/HeartCodec-oss-20260123 --local-dir ComfyUI/models/MuLaCover/HeartCodec-oss

（3）风格标签编码器
    https://huggingface.co/Qwen/Qwen3-Embedding-0.6B

    hf download Qwen/Qwen3-Embedding-0.6B --local-dir ComfyUI/models/MuLaCover/Qwen3-Embedding-0.6B

（4）转谱模型（只用 MIDI 模式可跳过）

YourMT3 转谱权重
    https://huggingface.co/spaces/mimbres/YourMT3

    mkdir -p ComfyUI/models/MuLaCover/SymbolicTranscriptor/yourmt3
    curl -fL 'https://huggingface.co/spaces/mimbres/YourMT3/resolve/main/amt/logs/2024/mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/checkpoints/last.ckpt' -o 'ComfyUI/models/MuLaCover/SymbolicTranscriptor/yourmt3/last.ckpt'

五个 ChordNet 和弦模型
    https://github.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition/tree/master/cache_data

bash 版：

    cd ComfyUI/models/MuLaCover/SymbolicTranscriptor
    mkdir -p chord
    for fold in 0 1 2 3 4; do
      f="joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s${fold}.best.sdict"
      curl -fL "https://raw.githubusercontent.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition/master/cache_data/$f" -o "chord/$f"
    done

PowerShell 版：

    cd ComfyUI\models\MuLaCover\SymbolicTranscriptor
    mkdir chord -Force
    0..4 | ForEach-Object {
      $f = "joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s$_.best.sdict"
      curl.exe -fL "https://raw.githubusercontent.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition/master/cache_data/$f" -o "chord\$f"
    }


【六、使用方法】

参考音频模式：
  Load Audio 的 audio 接生成节点 ref_audio，模型加载的 pipe、风格标签的 tags
  一并接入「MuLaCover 生成」，lyrics 框填歌词，输出 audio。
  save_symbolic 开启时，转谱产物（melody/chord/drums 三个 MIDI + bpm.txt）
  输出到 ComfyUI/output/<symbolic_output>/symbolic_时间戳/

MIDI 模式：
  「MuLaCover 加载 MIDI」两个实例分别接 melody_midi 与 chord_midi
  （必须同时连接，drum_midi 可选），加 pipe 进「MuLaCover 生成」。
  此模式不加载转谱模型，启动更快、内存占用更小。

推荐工作流（转谱产物复用）：
  1. 换新参考曲 → 参考音频模式跑一次（付一次转谱模型加载成本），
     开启 save_symbolic 导出转谱 MIDI
  2. 之后同一首歌反复改歌词/标签迭代 → 把导出的 melody/chord MIDI
     用上传按钮喂回「加载 MIDI」节点，改走 MIDI 模式——更快，
     还可以在 MIDI 编辑器里手改旋律、和弦后再生成


【七、参数说明】

max_audio_length —— 生成时长上限（秒，10~480）；模型唱完歌词会提前结束
temperature / topk / cfg_scale —— 采样参数；默认 1.0 / 250 / 1.5，一般不用动
seed —— 支持固定 / 随机 / 递增
bpm —— 仅参考音频模式有效；0 = 自动检测
keep_in_ram（加载节点）—— 模型阶段间隙驻留系统内存，二次生成数秒恢复；
    内存不足则关闭（每次从磁盘重载约 1-2 分钟）
unload_after_generate（生成节点）—— 开启则每次生成完彻底释放显存，
    优先级高于 keep_in_ram
save_symbolic / symbolic_output —— 参考音频模式导出转谱 MIDI 到 output 下指定目录


【八、常见问题】

· 加载 MIDI 节点没有上传按钮：浏览器 Ctrl+F5 强刷；仍无则看 F12 控制台
  是否有 mulacover.js 加载失败
· 手动拷了 MIDI 到 input 目录但下拉框没有：刷新页面即可，无需重启 ComfyUI
· 系统内存吃紧：关闭 keep_in_ram（回到官方行为，每次磁盘重载）；或改用 MIDI 模式
· 显存不足：调小 max_audio_length，或开启 unload_after_generate
· ref_audio 与 MIDI 报互斥：官方约束，两组输入只能接一组
· 生成时长总是远小于设定值：模型按歌词唱完自动停止，属正常；想要更长就加长歌词


【九、许可】

· 本插件代码：Apache-2.0
· MuLaCover 官方权重与生成输出：CC BY-NC 4.0 + MODEL_LICENSE，仅限非商用
  详见 https://github.com/HeartMuLa/MuLaCover/blob/main/LICENSING.md


【十、致谢】

· HeartMuLa/MuLaCover —— 模型与推理包（本仓库 MuLaCover/ 目录为其源码副本）
· mimbres/YourMT3 —— 音频转谱
· music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition —— 和弦识别
· Qwen3-Embedding —— 风格标签编码
