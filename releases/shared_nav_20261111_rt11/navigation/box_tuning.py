"""Operator-facing units are centimetres and centimetres per second."""
from dataclasses import asdict, dataclass
import math

# field, label, unit, minimum, maximum, slider increment
FIELDS = (
    ('side_step', '横移一步', 'cm', 1., 9., .1),
    ('forward_step', '前进一步', 'cm', 1., 12., .1),
    ('side_speed', '横移速度上限', 'cm/s', 10., 30., 1.),
    ('forward_speed', '前进速度上限', 'cm/s', 20., 30., 1.),
    ('first_gap', '首箱横向让开距离', 'cm', 36., 45., .5),
)

DEPTH_REASON_TEXT = {
    'missing_ground_plane': '等待深度相机测出地面',
    'missing_body_geometry': '等待与画面时间匹配的手脚位置',
    'missing_capture_body': '拍照时缺少手脚位置，等待下一帧',
    'stale_capture_body': '拍照时的关节反馈已过期',
    'interrupted_body_history': '拍照后的关节反馈有间断，等待重新确认',
    'stale_live_body': '迈步前的手脚位置尚未更新',
    'insufficient_floor': '前方可测到的地面太少',
    'missing_floor_band': '深度画面尚未覆盖车身两侧的地面',
    'inconsistent_corridor': '两侧测量不一致，等待新画面',
    'unseen_swept_path': '下一步手脚经过的位置仍有深度盲区',
    'obstacle_in_step': '下一步可能碰到障碍，重新观察',
    'invalid_step_length': '前进一步的距离超出范围',
}


@dataclass(frozen=True)
class BoxTuning:
    side_step: float = 4.6
    forward_step: float = 6.3
    side_speed: float = 26.
    forward_speed: float = 22.
    first_gap: float = 39.

    @classmethod
    def read(cls, values):
        if not isinstance(values,dict) or set(values)!=set(asdict(cls())):
            raise ValueError('请提交完整的五项设置')
        for key,label,unit,lo,hi,increment in FIELDS:
            value=values[key]
            if (type(value) not in (int,float) or not math.isfinite(value) or
                    not lo<=value<=hi or abs(round((value-lo)/increment)*increment+lo-value)>1e-6):
                raise ValueError(f'{label}允许 {lo:g}–{hi:g} {unit}，步进 {increment:g}')
        return cls(**values)
