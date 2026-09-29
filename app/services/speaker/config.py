"""Internal voiceprint constants."""

VOICEPRINT_PROVIDER = "campplus"
VOICEPRINT_MATCH_MARGIN = 0.08
VOICEPRINT_TOP_K = 5
VOICEPRINT_MIN_SPEECH_SEC = 2.0
# 单次送入声纹模型的音频上限（秒）。CAM++ 期望短时输入：把数小时拼接音频
# 整体送入会让 TDNN 中间激活爆显存，且长时拼接的单一 embedding 也不具代表性。
VOICEPRINT_MAX_CHUNK_SEC = 30.0
# 尾片短于此值则丢弃（太短提取出的 embedding 不可靠）
VOICEPRINT_MIN_CHUNK_SEC = 1.0
# 每个说话人最多参与融合的分段数（0=不限制）。长对话可能有上百段，按本值
# 等间隔采样：centroid embedding 用几十段估计已足够稳定，可明显缩短提取耗时。
VOICEPRINT_MAX_SEGMENTS_PER_SPEAKER = 30
VOICEPRINT_AUTO_CREATE_SCHEMA = True
