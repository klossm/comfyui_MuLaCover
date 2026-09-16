import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import gc
import json
import time
import uuid
import traceback
from pathlib import Path

import folder_paths

MULACOVER_MODEL_DIR = Path(
    os.environ.get("MULACOVER_MODEL_ROOT",
                   os.path.join(folder_paths.models_dir, "mulacover"))
).resolve()
folder_paths.add_model_folder_path("mulacover", str(MULACOVER_MODEL_DIR))

_HERE = Path(__file__).resolve().parent
_REPO = _HERE / "MuLaCover"
for _base in (_REPO, _REPO / "src", _HERE):
    _pkg = _base / "mulacover"
    if _pkg.is_dir() and ((_pkg / "__init__.py").exists() or any(_pkg.glob("*.py"))):
        if str(_base) not in sys.path:
            sys.path.insert(0, str(_base))
        print(f"[MuLaCover] 使用本地包: {_pkg}")
        break


def _fix_numpy_compat():
    try:
        import numpy as np
        fake = getattr(np, "_core", None)
        if fake is not None and not hasattr(fake, "multiarray"):
            np._core = np.core
            sys.modules["numpy._core"] = np.core
            sys.modules["numpy._core.multiarray"] = np.core.multiarray
            print("[MuLaCover] numpy 修复: 假 _core 已重定向到 np.core")
    except Exception as e:
        print(f"[MuLaCover] numpy 修复检查异常: {e}")


_fix_numpy_compat()

try:
    import torch
    import numpy as np
    import soundfile as sf
    from mulacover import MuLaCoverGenPipeline
    HAS_MULACOVER = True
    IMPORT_ERROR = ""
except Exception:
    HAS_MULACOVER = False
    IMPORT_ERROR = traceback.format_exc()


# ── 用户中断响应：点 ComfyUI Cancel 立刻终止采样 ──────────────────────
def _check_interrupt():
    """每步回调都检查一次（内部只是读标志位，开销可忽略）。"""
    try:
        import comfy.model_management as _mm
        _mm.throw_exception_if_processing_interrupted()
    except ImportError:
        pass


# ── 显存协调（借鉴 ComfyUI-YuE2 的 free_yue2_first 模式）──────────────
# 我们的管线是裸 torch 加载，绕过了 ComfyUI 的模型管理——comfy 在显存
# 不足时只会为"自己管理的模型"做腾退，不会为我们腾。混用工作流
# （SD 出参考图/人声分离后接 MuLaCover）时，先手动请 comfy 卸载它的
# 模型，12GB 卡才装得下 HeartMuLa-3B。后续节点再用到 SD 时 comfy
# 会自行重载（代价是几秒加载时间），无功能损失。
def _free_comfy_models():
    try:
        import comfy.model_management as mm
        mm.unload_all_models()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("[MuLaCover] 已请 ComfyUI 卸载其管理的模型，为生成腾出显存")
    except Exception as e:
        print(f"[MuLaCover][WARN] 释放 ComfyUI 模型失败（忽略，继续生成）: {e}")


# ── TF32 快速转谱（借鉴 ComfyUI-YuE2 的 runtime_flags 存/恢复模式）────
# YourMT3 + ChordNet 转谱全程 fp32；Ampere 及以上显卡开 TF32 后 matmul
# 约 2~4 倍提速，对"输出音符/和弦符号"这类粗粒度结果无实际影响。
# 主生成模型是 bf16（TF32 不参与），HeartCodec 解码不包在开关内，
# 音质路径完全不受影响。非 Ampere 卡上 TF32 是空操作，无副作用。
def _wrap_fast_fp32(pipe):
    """给 _symbolic_condition 套 TF32 开关（幂等）；是否生效由
    pipe._mc_fast_fp32 决定，生成节点每次运行前设置。"""
    if getattr(pipe, "_mc_fast_fp32_wrapped", False):
        return
    orig = pipe._symbolic_condition

    def symcond_fast(*args, **kw):
        if not getattr(pipe, "_mc_fast_fp32", False):
            return orig(*args, **kw)
        saved = (torch.backends.cuda.matmul.allow_tf32,
                 torch.backends.cudnn.allow_tf32)
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("[MuLaCover] 转谱阶段 TF32 加速已启用")
            return orig(*args, **kw)
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved

    pipe._symbolic_condition = symcond_fast
    pipe._mc_fast_fp32_wrapped = True


# ── 进度桥接：拦截 mulacover 内部 tqdm → 节点进度条 + 控制台单行刷新 ──
_PBAR_HOOK = {"pbar": None, "seg_start": 10, "seg_end": 90,
              "cur": 10, "last_push": 0.0, "line_open": False}


def _push_progress(total, n):
    _check_interrupt()  # 必须在一切早退 return 之前，保证每步都检查
    hook = _PBAR_HOOK
    pbar = hook["pbar"]
    if pbar is None or not total or not n:
        return
    span = hook["seg_end"] - hook["seg_start"]
    pct = hook["seg_start"] + span * min(n / float(total), 1.0)
    pct = max(pct, hook["cur"])
    now = time.time()
    if pct - hook["cur"] >= 1.0 or (pct > hook["cur"] and now - hook["last_push"] > 0.5):
        hook["cur"] = pct
        hook["last_push"] = now
        try:
            pbar.update_absolute(int(pct))
        except Exception:
            pass


class _HookedTqdm:
    def __init__(self, iterable=None, *args, **kwargs):
        self._iterable = iterable
        self._desc = str(kwargs.get("desc") or kwargs.get("desc_str") or "").strip()
        total = kwargs.get("total")
        if total is None and iterable is not None:
            try:
                total = len(iterable)
            except Exception:
                total = None
        self._total = int(total) if total else 0
        self._n = 0
        self._last_log = 0.0
        self._t0 = None
        self.disable = bool(kwargs.get("disable", False))

    def __iter__(self):
        if self._iterable is None:
            return
        for item in self._iterable:
            yield item
            self.update(1)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def _render(self, done):
        if self._t0 is None:
            self._t0 = time.time()
        label = self._desc or "推理"
        if not self._total:
            print(f"\r[MuLaCover] {label} {self._n} 步 ", end="", flush=True)
            _PBAR_HOOK["line_open"] = True
            return
        pct = 100.0 * self._n / self._total
        rate = self._n / max(time.time() - self._t0, 1e-6)
        eta = (self._total - self._n) / rate if rate > 0 else 0.0
        eta_str = f"{eta:.0f}秒" if eta < 60 else f"{eta / 60:.1f}分"
        line = (f"\r[MuLaCover] {label} {self._n}/{self._total} ({pct:.0f}%) "
                f"| {rate:.1f} it/s | 剩余~{eta_str} ")
        if done:
            print(line, flush=True)
            _PBAR_HOOK["line_open"] = False
        else:
            print(line, end="", flush=True)
            _PBAR_HOOK["line_open"] = True

    def update(self, n=1):
        if self.disable:
            return
        self._n += n
        _push_progress(self._total, self._n)
        now = time.time()
        if self._total:
            done = self._n >= self._total
            if done or now - self._last_log >= 0.5:
                self._last_log = now
                self._render(done)
        elif now - self._last_log >= 1.0:
            self._last_log = now
            self._render(False)

    def close(self): pass
    def refresh(self): pass
    def clear(self): pass
    def set_description(self, *a, **k): pass
    def set_description_str(self, *a, **k): pass
    def set_postfix(self, *a, **k): pass
    def set_postfix_str(self, *a, **k): pass
    def write(self, *a, **k): pass
    def moveto(self, *a, **k): pass

    def reset(self, total=None):
        self._n = 0
        self._t0 = None
        if total is not None:
            self._total = int(total)

    @property
    def n(self):
        return self._n

    @property
    def total(self):
        return self._total


def _install_tqdm_hooks():
    hooked = []
    for mod_name in ("mulacover.pipeline", "mulacover._codec.modeling",
                     "mulacover._symbolic_transcription"):
        try:
            __import__(mod_name)
        except Exception:
            pass
    for mod_name, mod in list(sys.modules.items()):
        if not mod_name.startswith("mulacover") or mod is None:
            continue
        if hasattr(mod, "tqdm") and getattr(mod, "tqdm") is not _HookedTqdm:
            hooked.append((mod, getattr(mod, "tqdm")))
            mod.tqdm = _HookedTqdm
    if hooked:
        print(f"[MuLaCover] 进度桥接已安装到 {len(hooked)} 个模块命名空间")
    else:
        print("[MuLaCover][WARN] 未找到 mulacover 内部 tqdm，进度将退化为分段跳变")
    return hooked


def _restore_tqdm_hooks(hooked):
    for mod, orig in hooked:
        try:
            mod.tqdm = orig
        except Exception:
            pass
    if _PBAR_HOOK.get("line_open"):
        print(flush=True)
        _PBAR_HOOK["line_open"] = False
    _PBAR_HOOK["pbar"] = None


# ── 模型完整性校验 ────────────────────────────────────────────────────
# 利用 safetensors 格式自描述特性精确校验：8B 头长 + header JSON + 数据区
# 三者之和必须等于文件实际大小，能报出"差多少字节"，对任何版本权重有效。

def _check_safetensors(path: Path):
    """完整返回 None，否则返回问题字符串（含精确字节数差异）"""
    try:
        size = path.stat().st_size
        if size < 8:
            return f"文件过小（{size}B，疑似空文件或下载中断）"
        with open(path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            if n <= 0 or n > 256 * 1024 * 1024:
                return f"头部长度异常（{n}B），文件已损坏"
            if size < 8 + n:
                return f"文件被截断：头区需 {8 + n}B，实际仅 {size}B，请重新下载"
            header = json.loads(f.read(n))
        if not isinstance(header, dict):
            return "header 格式异常"
        ends = [v["data_offsets"][1] for v in header.values()
                if isinstance(v, dict) and "data_offsets" in v]
        if ends:
            expected = 8 + n + max(ends)
            if size != expected:
                missing = expected - size
                return (f"数据区不完整：应为 {expected}B，实际 {size}B"
                        f"（差 {missing}B），请重新下载该分片")
        return None
    except PermissionError:
        return "无读取权限（可能被杀毒软件或占用中）"
    except Exception as e:
        return f"无法解析（{type(e).__name__}: {e}）"


def _check_main_model(root: Path, model_sub: Path):
    """主生成模型：index.json 的 weight_map 逐分片精确校验"""
    problems = []
    idx = model_sub / "model.safetensors.index.json"
    if not idx.is_file():
        problems.append(f"{model_sub.name}/ 缺少 model.safetensors.index.json")
    else:
        try:
            wm = json.loads(idx.read_text(encoding="utf-8")).get("weight_map") or {}
            shards = sorted(set(wm.values()))
            if not shards:
                problems.append("index.json 里没有 weight_map，文件可能下载有误")
            for sh in shards:
                p = model_sub / sh
                if not p.is_file():
                    problems.append(f"{model_sub.name}/{sh} 缺失")
                else:
                    err = _check_safetensors(p)
                    if err:
                        problems.append(f"{model_sub.name}/{sh}: {err}")
            if not problems:
                print(f"[MuLaCover] 主模型完整性校验：{len(shards)} 个分片全部通过")
        except Exception as e:
            problems.append(f"index.json 解析失败: {e}")
    for fname in ("config.json", "tokenizer.json"):
        if not (model_sub / fname).is_file():
            problems.append(f"{model_sub.name}/{fname} 缺失")
    return problems


def _check_transcriptor(root: Path):
    """转谱模型检查 —— 仅参考音频模式调用；MIDI 模式完全不需要这些文件"""
    problems = []
    tr = root / "SymbolicTranscriptor"
    if not tr.is_dir():
        return ["缺少目录 SymbolicTranscriptor/（转谱模型整套缺失）"]
    if not (tr / "yourmt3" / "last.ckpt").is_file():
        problems.append("缺少 SymbolicTranscriptor/yourmt3/last.ckpt（YourMT3 转谱权重，约 2.4GB）")
    chord_dir = tr / "chord"
    if not chord_dir.is_dir():
        problems.append("缺少 SymbolicTranscriptor/chord/ 目录（和弦识别模型）")
    else:
        n = len(list(chord_dir.glob("*.sdict")))
        if n < 5:
            problems.append(f"chord/ 只找到 {n} 个 .sdict，需要 5 个（ChordNet 五折模型）")
    return problems


# ── 管线缓存 ──────────────────────────────────────────────────────────
_PIPES = {}
_PIPE_KEYS = {}


def _register_pipe(pipe, key):
    _PIPES[key] = pipe
    _PIPE_KEYS[id(pipe)] = key


def _drop_pipe_models(pipe):
    for name in ("mulacover", "codec", "qwen", "transcriptor"):
        setattr(pipe, f"_{name}", None)
    pipe._qwen_tokenizer = None
    pipe._cache_batch_size = None


def _unload_pipe(pipe=None):
    removed = 0
    try:
        if pipe is not None:
            _drop_pipe_models(pipe)
            key = _PIPE_KEYS.pop(id(pipe), None)
            if key is not None and key in _PIPES:
                del _PIPES[key]
            removed = 1
        else:
            for p in list(_PIPES.values()):
                try:
                    _drop_pipe_models(p)
                except Exception:
                    pass
            removed = len(_PIPES)
            _PIPES.clear()
            _PIPE_KEYS.clear()
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return removed


# ── 常驻模式：官方 _release 在 lazy_load=True 时丢弃模型（磁盘重载），
# 这里改为转存 CPU 内存，各阶段入口搬回 GPU ─────────────────────────

def _remove_accel_hooks(module):
    try:
        from accelerate.hooks import remove_hook_from_module
        remove_hook_from_module(module, recurse=True)
    except Exception:
        pass


def _collect_modules(obj, depth=0, seen=None, out=None):
    if seen is None:
        seen, out = set(), []
    if depth > 4 or id(obj) in seen:
        return out
    seen.add(id(obj))
    if isinstance(obj, torch.nn.Module):
        out.append(obj)
        return out
    if isinstance(obj, (list, tuple, set, frozenset)):
        for v in obj:
            _collect_modules(v, depth + 1, seen, out)
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_modules(v, depth + 1, seen, out)
    else:
        d = getattr(obj, "__dict__", None)
        if isinstance(d, dict):
            for v in list(d.values()):
                _collect_modules(v, depth + 1, seen, out)
    return out


def _drop_by_name(pipe, name):
    setattr(pipe, f"_{name}", None)
    if name == "qwen":
        pipe._qwen_tokenizer = None
    if name == "mulacover":
        pipe._cache_batch_size = None


def _enable_resident_mode(pipe):
    if getattr(pipe, "_resident_mode", False):
        return
    orig_release = pipe._release
    orig_forward = pipe._forward
    orig_postprocess = pipe.postprocess
    orig_load_qwen = pipe._load_qwen
    orig_symcond = pipe._symbolic_condition

    def _to_cpu(name):
        obj = getattr(pipe, f"_{name}", None)
        if obj is None:
            return
        mods = _collect_modules(obj)
        if not mods:
            print(f"[MuLaCover][WARN] {name} 内部未找到 nn.Module，按官方行为丢弃")
            _drop_by_name(pipe, name)
            return
        try:
            for m in mods:
                _remove_accel_hooks(m)
                m.to("cpu")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[MuLaCover] {name} 已转存CPU（{len(mods)} 个模块），显存已释放")
        except Exception as e:
            print(f"[MuLaCover][WARN] {name} 转存CPU失败"
                  f"({type(e).__name__}: {e})，按官方行为丢弃")
            _drop_by_name(pipe, name)

    def _restore(name):
        obj = getattr(pipe, f"_{name}", None)
        if obj is None:
            return
        target = pipe.devices[name]
        mods = _collect_modules(obj)
        if not mods:
            return
        t0 = time.time()
        try:
            for m in mods:
                p = next(m.parameters(), None)
                if p is not None and p.device != target:
                    m.to(target)
            print(f"[MuLaCover] {name} 已从CPU恢复GPU ({time.time() - t0:.1f}s)")
        except Exception as e:
            print(f"[MuLaCover][WARN] {name} 恢复GPU失败: {e}")

    def fwd(model_inputs, **kw):
        _restore("mulacover")
        return orig_forward(model_inputs, **kw)

    def post(model_outputs, **kw):
        _restore("codec")
        return orig_postprocess(model_outputs, **kw)

    def lq():
        orig_load_qwen()
        _restore("qwen")

    def symcond(*args, **kw):
        _restore("transcriptor")
        return orig_symcond(*args, **kw)

    pipe._release = _to_cpu
    pipe._forward = fwd
    pipe.postprocess = post
    pipe._load_qwen = lq
    pipe._symbolic_condition = symcond
    pipe._resident_mode = True
    print("[MuLaCover] 常驻模式已启用：模型在阶段间隙转存CPU内存，"
          "恢复GPU仅需数秒（首次运行仍需从磁盘加载一次）")


# ── 通用工具 ──────────────────────────────────────────────────────────

def _to_int(v, default):
    try:
        s = str(v).strip()
        return int(float(s)) if s else default
    except (TypeError, ValueError):
        return default


def _resolve_input_file(name_or_path):
    p = (name_or_path or "").strip().strip('"').strip("'")
    if not p:
        return None
    path = Path(p).expanduser()
    if path.is_absolute():
        return path if path.is_file() else None
    for base in (folder_paths.get_input_directory(),
                 folder_paths.get_output_directory()):
        cand = Path(base) / path
        if cand.is_file():
            return cand.resolve()
    return None


def _audio_to_temp_wav(audio, prefix="ref"):
    wf = audio["waveform"]
    sr = int(audio["sample_rate"])
    if not isinstance(wf, torch.Tensor):
        wf = torch.tensor(wf)
    wf = wf.detach().float().cpu()
    if wf.ndim == 3:
        wf = wf[0]
    if wf.ndim == 1:
        wf = wf.unsqueeze(0)
    data = wf.numpy().T
    tmp_dir = Path(folder_paths.get_temp_directory()) / "mulacover"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / f"{prefix}_{uuid.uuid4().hex}.wav"
    sf.write(str(path), data, sr, subtype="FLOAT")
    return path


def _midi_path(v):
    if v is None:
        return None
    if isinstance(v, dict):
        v = v.get("path") or v.get("midi_path") or v.get("value")
        if isinstance(v, (list, tuple)):
            v = v[0] if v else None
    if v is None:
        return None
    return _resolve_input_file(str(v))


# ── 风格标签清洗 ──────────────────────────────────────────────────────

def _clean_style_field(value):
    v = str(value or "")
    v = v.replace(";", ",").replace("[", "(").replace("]", ")")
    return " ".join(v.split())


def _validate_tags_string(tags):
    """用户在生成节点直接手贴整串 tags 时的括号平衡校验"""
    depth = 0
    for i, ch in enumerate(str(tags)):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth < 0:
                raise ValueError(
                    f"tags 第 {i + 1} 个字符出现多余的 ']' —— "
                    f"字段值内不能使用方括号（会被当成字段结束），请改用圆括号")
    if depth != 0:
        raise ValueError("tags 方括号不配对：'[' 比 ']' 多，请检查每一段 field:[value]")


# ── 节点：模型加载 ────────────────────────────────────────────────────

class MuLaCoverLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (["cuda:0", "cuda:1", "cpu"],),
                "main_dtype": (["bfloat16", "float16", "float32"],
                               {"default": "bfloat16"}),
                "keep_in_ram": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "开启：模型在阶段间隙转存CPU内存，二次生成数秒恢复"
                               "（系统内存需≥24GB）；关闭：每次从磁盘重新加载"}),
            }
        }

    RETURN_TYPES = ("MULACOVER_PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "load"
    CATEGORY = "MuLaCover"

    def load(self, device, main_dtype, keep_in_ram):
        if not HAS_MULACOVER:
            raise RuntimeError(f"导入 mulacover 失败：\n{IMPORT_ERROR}")
        root = MULACOVER_MODEL_DIR
        if not root.is_dir():
            raise FileNotFoundError(
                f"模型目录不存在: {root}\n"
                f"布局: MuLaCover\\ / HeartCodec-oss\\ / "
                f"Qwen3-Embedding-0.6B\\ / SymbolicTranscriptor\\")

        model_sub = root / "MuLaCover"
        if not model_sub.is_dir():
            model_sub = root / "MuLaCover-oss"
        if not model_sub.is_dir():
            raise FileNotFoundError(f"找不到 MuLaCover\\ 或 MuLaCover-oss\\: {root}")

        problems = _check_main_model(root, model_sub)
        for sub in ("HeartCodec-oss", "Qwen3-Embedding-0.6B"):
            if not (root / sub).is_dir():
                problems.append(f"缺少目录 {sub}/")
        if problems:
            raise FileNotFoundError(
                f"模型文件不完整（{len(problems)} 个问题）：\n- "
                + "\n- ".join(problems)
                + "\n\n请对照插件 README『模型下载』一节补齐；"
                  "重跑同一条 hf download 命令即可断点续传补齐分片")

        dt = {"bfloat16": torch.bfloat16,
              "float16": torch.float16,
              "float32": torch.float32}[main_dtype]

        key = (str(root), device, main_dtype, bool(keep_in_ram))
        if key in _PIPES:
            return (_PIPES[key],)

        print(f"[MuLaCover] 加载管线: {root} @ {device} ({main_dtype})")
        pipe = MuLaCoverGenPipeline.from_pretrained(
            str(root),
            device=torch.device(device),
            dtype={"mulacover": dt, "codec": torch.float32,
                   "qwen": torch.float32, "transcriptor": torch.float32},
            lazy_load=True,
        )
        if keep_in_ram and device.startswith("cuda"):
            _enable_resident_mode(pipe)
        _wrap_fast_fp32(pipe)  # TF32 开关壳（是否生效由生成节点每次设置）
        _register_pipe(pipe, key)
        return (pipe,)


# ── 节点：风格标签 ────────────────────────────────────────────────────

class MuLaCoverStyleTags:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "topic": ("STRING", {"default": "heartbreak and longing"}),
                "genre": ("STRING", {"default": "psychedelic synthwave, city pop"}),
                "instrument": ("STRING", {"default": "synth bass, gated reverb drums, "
                                                     "electric guitar, Rhodes piano, "
                                                     "muted guitar"}),
                "mood": ("STRING", {"default": "melancholic, explosive, grand"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tags",)
    FUNCTION = "build"
    CATEGORY = "MuLaCover"

    def build(self, topic, genre, instrument, mood):
        topic = _clean_style_field(topic)
        genre = _clean_style_field(genre)
        instrument = _clean_style_field(instrument)
        mood = _clean_style_field(mood)
        if not any((topic, genre, instrument, mood)):
            raise ValueError("风格标签四个字段全为空，至少填写一个字段")
        parts = []
        for name, v in (("topic", topic), ("genre", genre),
                        ("instrument", instrument), ("mood", mood)):
            if not v:
                print(f"[MuLaCover][WARN] 风格字段 {name} 为空，已自动填入 unspecified")
                v = "unspecified"
            parts.append(f"{name}:[{v}]")
        return ("; ".join(parts),)


# ── 节点：MIDI 加载（下拉选择 + 上传按钮，机制同官方 LoadAudio）──────

class MuLaCoverLoadMIDI:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = []
        if os.path.isdir(input_dir):
            for dirpath, _, filenames in os.walk(input_dir):
                for name in filenames:
                    if name.lower().endswith((".mid", ".midi")):
                        rel = os.path.relpath(os.path.join(dirpath, name), input_dir)
                        files.append(rel.replace("\\", "/"))
        options = sorted(files) or ["(input目录没有MIDI，请点上传按钮)"]
        return {"required": {
            "midi_file": (options, {"midi_upload": True}),
        }}

    RETURN_TYPES = ("MIDI",)
    RETURN_NAMES = ("midi",)
    FUNCTION = "load"
    CATEGORY = "MuLaCover"

    def load(self, midi_file):
        path = Path(folder_paths.get_annotated_filepath(midi_file))
        if not path.is_file():
            raise FileNotFoundError(
                f"找不到 MIDI 文件: {path}\n"
                f"请用上传按钮选择文件，或放到 ComfyUI/input/ 后刷新页面")
        return (str(path),)


# ── 节点：生成 ────────────────────────────────────────────────────────

class MuLaCoverGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("MULACOVER_PIPE",),
                "lyrics": ("STRING", {
                    "multiline": True,
                    "default": "[Verse]\n歌词第一行\n\n[Chorus]\n副歌歌词"}),
                "tags": ("STRING", {
                    "multiline": True,
                    "default": "topic:[longing]; genre:[contemporary R&B, neo soul]; "
                               "instrument:[808 bass, Rhodes piano, muted guitar, "
                               "female vocal]; mood:[sultry, soulful, yearning]"}),
                "seed": ("INT", {"default": 42, "min": 0,
                                 "max": 0xFFFFFFFFFFFFFFFF,
                                 "control_after_generate": True}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.01,
                                          "max": 2.0, "step": 0.01}),
                "topk": ("INT", {"default": 250, "min": 1, "max": 8191,
                                 "tooltip": "官方合法范围 1~8191"}),
                "cfg_scale": ("FLOAT", {"default": 1.5, "min": 1.0,
                                        "max": 5.0, "step": 0.1}),
                "max_audio_length": ("FLOAT", {
                    "default": 300.0, "min": 10.0, "max": 480.0, "step": 1.0,
                    "tooltip": "生成时长（秒），范围 10~480"}),
                "bpm": ("INT", {"default": 0, "min": 0, "max": 300,
                                "tooltip": "仅参考音频模式有效；0=自动检测"}),
                "free_vram_before": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "生成前请 ComfyUI 卸载其管理的模型（SD 等），为 "
                               "HeartMuLa-3B 腾显存。我们的管线绕过 comfy 模型管理，"
                               "混用工作流时 comfy 不会自动腾退；12GB 卡建议开启。"
                               "后续节点再用到那些模型会自动重载（多几秒）"}),
                "fast_fp32_transcribe": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "参考音频模式的转谱/和弦识别阶段启用 TF32 加速"
                               "（Ampere 及以上约 2~4 倍）。不影响主模型与音质；"
                               "需逐位复现官方结果时关闭"}),
                "symbolic_output": ("STRING", {
                    "default": "mulacover",
                    "tooltip": "转谱产物输出目录：相对 ComfyUI/output，"
                               "实际落盘在其下 symbolic_时间戳/ 子目录"}),
                "save_symbolic": ("BOOLEAN", {"default": True}),
                "unload_after_generate": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "开启：每次生成完彻底释放（下次从磁盘重载约1-2分钟）；"
                               "关闭：配合 Loader 的 keep_in_ram，模型驻留CPU内存，"
                               "二次生成数秒恢复"}),
            },
            "optional": {
                "ref_audio": ("AUDIO",),
                "melody_midi": ("MIDI",),
                "chord_midi": ("MIDI",),
                "drum_midi": ("MIDI",),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "symbolic_dir")
    FUNCTION = "generate"
    CATEGORY = "MuLaCover"
    OUTPUT_NODE = True

    def generate(self, pipe, lyrics, tags, seed, temperature, topk, cfg_scale,
                 max_audio_length, bpm, free_vram_before, fast_fp32_transcribe,
                 symbolic_output, save_symbolic, unload_after_generate,
                 ref_audio=None, melody_midi=None, chord_midi=None, drum_midi=None):
        _fix_numpy_compat()

        _validate_tags_string(tags)

        pbar = None
        try:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(100)
            pbar.update_absolute(2)
        except Exception:
            pbar = None
        _PBAR_HOOK["pbar"] = pbar
        _PBAR_HOOK["cur"] = 10
        _PBAR_HOOK["last_push"] = time.time()

        seconds = max(10.0, min(float(max_audio_length or 300.0), 480.0))
        max_ms = max(80, min(int(round(seconds * 1000)), 480000))
        topk = _to_int(topk, 250)
        seed = _to_int(seed, 0)
        bpm = _to_int(bpm, 0)
        cfg_scale = float(cfg_scale or 1.5)
        temperature = float(temperature or 1.0)

        has_audio = ref_audio is not None
        has_mel = melody_midi is not None
        has_chd = chord_midi is not None
        has_drm = drum_midi is not None

        if has_audio and (has_mel or has_chd or has_drm):
            raise ValueError("ref_audio 与 MIDI 输入互斥（官方约束）")
        if has_audio:
            mode = "reference_audio"
            tr_problems = _check_transcriptor(MULACOVER_MODEL_DIR)
            if tr_problems:
                raise FileNotFoundError(
                    "参考音频模式需要转谱模型，当前缺失：\n- "
                    + "\n- ".join(tr_problems)
                    + "\n\n（只用 MIDI 模式则无需这些文件，可忽略本错误）")
        elif has_mel or has_chd or has_drm:
            mode = "midi"
            if not (has_mel and has_chd):
                raise ValueError("MIDI 模式必须同时连接 melody_midi 与 chord_midi")
        else:
            raise ValueError("请连接 ref_audio(AUDIO) 或 melody/chord(MIDI) 输入端")

        # TF32 开关：转谱壳每次调用时读取此标志（MIDI 模式不进转谱，无影响）
        try:
            pipe._mc_fast_fp32 = bool(fast_fp32_transcribe)
        except Exception:
            pass

        # 显存协调：在真正需要显存前请 comfy 腾地方（借鉴 YuE2 free_yue2_first）
        if free_vram_before and mode == "reference_audio":
            _free_comfy_models()

        cond = {"lyrics": lyrics, "tags": tags}
        ref_tmp = None
        if mode == "reference_audio":
            ref_tmp = _audio_to_temp_wav(ref_audio, prefix="ref")
            cond["ref_audio"] = str(ref_tmp)
            if bpm > 0:
                cond["bpm"] = bpm
        else:
            mel = _midi_path(melody_midi)
            chd = _midi_path(chord_midi)
            drm = _midi_path(drum_midi)
            if mel is None or chd is None:
                raise FileNotFoundError(f"melody/chord MIDI 文件不存在\n"
                                        f"melody: {mel}\nchord: {chd}")
            cond["melody_midi"] = str(mel)
            cond["chord_midi"] = str(chd)
            if drm is not None:
                cond["drum_midi"] = str(drm)

        if pbar:
            pbar.update_absolute(5)

        out_dir = Path(folder_paths.get_temp_directory()) / "mulacover"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_wav = out_dir / f"mulacover_{time.strftime('%Y%m%d_%H%M%S')}_{seed % 100000}.wav"

        symbolic_dir_str = ""
        kwargs = dict(save_path=str(out_wav),
                      temperature=temperature,
                      topk=topk,
                      cfg_scale=cfg_scale,
                      max_audio_length_ms=max_ms,
                      disable_progress=False)
        if save_symbolic and mode == "reference_audio":
            sub = (symbolic_output or "mulacover").strip() or "mulacover"
            sym_root = Path(sub) if Path(sub).is_absolute() \
                else Path(folder_paths.get_output_directory()) / sub
            run_sym = sym_root / f"symbolic_{time.strftime('%Y%m%d_%H%M%S')}"
            run_sym.mkdir(parents=True, exist_ok=True)
            kwargs["symbolic_save_dir"] = str(run_sym)
            symbolic_dir_str = str(run_sym)

        _devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        print(f"[MuLaCover] 开始生成 | mode={mode} | seed={seed} | 时长={seconds:g}s")
        t0 = time.time()
        hooked = _install_tqdm_hooks()
        try:
            with torch.no_grad():
                with torch.random.fork_rng(devices=_devices):
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                    try:
                        pipe(cond, **kwargs)
                    except Exception as e:
                        if "out of memory" not in str(e).lower():
                            raise
                        _check_interrupt()  # OOM 重试前确认用户没点取消
                        print("[MuLaCover] VRAM OOM，清理显存后重试...")
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        pipe(cond, **kwargs)
        finally:
            _restore_tqdm_hooks(hooked)

        if not out_wav.exists():
            raise RuntimeError(f"生成结束但未找到输出文件: {out_wav}")
        print(f"[MuLaCover] 完成，耗时 {time.time() - t0:.1f}s -> {out_wav}")

        if pbar:
            pbar.update_absolute(93)

        data, sr = sf.read(str(out_wav), dtype="float32")
        wav = torch.from_numpy(data)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        else:
            wav = wav.T.contiguous()
        audio = {"waveform": wav.unsqueeze(0), "sample_rate": sr}

        ui = {"audio": [{"filename": out_wav.name,
                         "subfolder": "mulacover", "type": "temp"}]}
        if ref_tmp is not None:
            try:
                ref_tmp.unlink()
            except Exception:
                pass

        if unload_after_generate:
            n = _unload_pipe(pipe)
            print(f"[MuLaCover] 生成完毕，已彻底释放模型（下次从磁盘重载）({n})")

        if pbar:
            pbar.update_absolute(100)
        return {"ui": ui, "result": (audio, symbolic_dir_str)}


NODE_CLASS_MAPPINGS = {
    "MuLaCoverLoader": MuLaCoverLoader,
    "MuLaCoverStyleTags": MuLaCoverStyleTags,
    "MuLaCoverLoadMIDI": MuLaCoverLoadMIDI,
    "MuLaCoverGenerate": MuLaCoverGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MuLaCoverLoader": "MuLaCover 模型加载",
    "MuLaCoverStyleTags": "MuLaCover 风格标签",
    "MuLaCoverLoadMIDI": "MuLaCover 加载 MIDI",
    "MuLaCoverGenerate": "MuLaCover 生成",
}
