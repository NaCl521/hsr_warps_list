import csv
import pathlib

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

RAW_SIZE = 160
PADDING = 22
SIZE = RAW_SIZE + PADDING * 2
MIDDLE = SIZE // 2
HEAD = int(MIDDLE * 0.7)
MARGIN = 16


def alpha_merge(im: np.ndarray, bg_color):
    b, g, r, a = cv2.split(im)
    alpha_mask = a.astype(float) / 255.0
    alpha_mask_inv = 1.0 - alpha_mask
    im = cv2.merge((b, g, r))
    for c in range(3):
        im[..., c] = im[..., c] * alpha_mask + bg_color[c] * alpha_mask_inv
    return im


def circle_mask(im: np.ndarray):
    mask = np.zeros_like(im)
    center = (im.shape[0] // 2, im.shape[1] // 2)
    radius = min(im.shape[0], im.shape[1]) // 2
    cv2.circle(mask, center, radius, (255,) * im.shape[2], -1)
    im = cv2.bitwise_and(im, mask)
    return im


def load_history(file: pathlib.Path):
    items: dict[str, list[str]] = {}
    preview_items: list[str] = []
    special_items: dict[str, list[str]] = {}
    with file.open() as f:
        reader = csv.DictReader(f)
        _flag = False
        for row in reader:
            if _flag:
                special_items[row["跃迁"]] = row["角色5"].split("、")
            elif row["跃迁"] != "预告":
                items[row["跃迁"]] = row["角色5"].split("、")
            else:
                preview_items = row["角色5"].split("、")
                _flag = True
                if preview_items == [""]:
                    preview_items = []
    return items, preview_items, special_items


def read_profile(file: str):
    im = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    assert im is not None
    im = circle_mask(im)
    return im


def get_pos_simple(data: dict[str, list[str]]):
    _version_count = 0
    versions: list[int] = []
    for v in data:
        if int(v[0]) > _version_count:
            _version_count += 1
            versions.append(0)
        elif int(v[2]) >= versions[-1]:
            versions[-1] += 1

    position: dict[str, list[list[int]]] = {}
    _version_count = -1
    for v, names in data.items():
        if v[-1] == "1":
            _version_count += 1
        for i in range(len(names)):
            if names[i] not in position:
                position[names[i]] = [[0]]
            if v[-1] == "1":
                w = MARGIN + SIZE * (3 - i) + PADDING
            elif v == "3.8.3":
                w = MARGIN + MIDDLE + SIZE * (6 + i) + PADDING
            else:
                w = MARGIN + MIDDLE + SIZE * (4 + i) + PADDING
            h = MARGIN + HEAD + SIZE * _version_count + PADDING
            position[names[i]].append([h, w])
    return position, versions


def get_pos_preview(preview_data: list[str]):
    pos_preview: dict[str, list[list[int]]] = {}
    if len(preview_data) > 0:
        h = HEAD + SIZE * 2 + PADDING
        w = MARGIN + MIDDLE + SIZE * 6 + PADDING
        for name in preview_data[:-1:2]:
            pos_preview[name] = [[1], [h, w]]
            h += SIZE

        name = preview_data[-1]
        pos_preview[name] = [[1], [h, w + SIZE // 2]]

        h = HEAD + SIZE * 2 + PADDING
        w += SIZE
        for name in preview_data[1::2]:
            pos_preview[name] = [[1], [h, w]]
            h += SIZE
    return pos_preview


def get_pos_special(special_data: dict[str, list[str]]):
    pos_special: dict[str, list[list[int]]] = {}
    # 联动处理·临时版
    for warp, names in special_data.items():
        h = HEAD + SIZE * 2 + PADDING
        w = MARGIN
        for name in names:
            pos_special[name] = [[2], [h, w]]
            w += SIZE
    return pos_special


def get_position(
    data: dict[str, list[str]],
    preview_data: list[str],
    special_data: dict[str, list[str]],
):
    positions, versions = get_pos_simple(data)
    positions.update(get_pos_preview(preview_data))
    positions.update(get_pos_special(special_data))
    return positions, versions


def background(
    bg_color: str,
    versions: list[int],
    preview: bool = True,
):
    font_size = MIDDLE * 0.35
    font = ImageFont.truetype("ARLRDBD.TTF", font_size)
    font_cn = ImageFont.truetype("SmileySans-Oblique.ttf", font_size)
    _size = (MARGIN * 2 + MIDDLE + SIZE * 8, MARGIN * 2 + HEAD + SIZE * sum(versions))
    im = Image.new("RGB", _size, "white")
    draw = ImageDraw.Draw(im)

    w, h = im.size[0] / 2, MARGIN + HEAD
    for x in range(len(versions)):
        for y in range(versions[x]):
            if y == 0:
                draw.rectangle((0, h, im.size[0] - 1, h + SIZE - 1), bg_color)
            draw.text((w, h + SIZE / 2), f"{x + 1}.{y}", "black", font, "mm")
            h += SIZE

    # 3.8.3
    _w, _h = im.size[0] - MARGIN - SIZE * 2, MARGIN + HEAD + SIZE * 23
    draw.rectangle((_w - 8, _h, _w + 8, h + SIZE - 1), "#ffd24d")
    size = (MARGIN + SIZE, _h + SIZE / 2)
    draw.text(size, "注：3.8 有三期跃迁", "black", font_cn, "mm")

    size = (im.size[0] / 2, MARGIN + HEAD / 2)
    draw.text(size, f"截 至 {len(versions)}.{versions[-1] - 1}", "black", font_cn, "mm")
    size = (MARGIN + SIZE * 2, MARGIN + HEAD / 2)
    draw.text(size, "开 拓 者 不 歪 小 保 底", "black", font_cn, "mm")
    size = (MARGIN + MIDDLE + SIZE * 6, MARGIN + HEAD / 2)
    draw.text(size, "开 拓 者 副 本 满 掉 落", "black", font_cn, "mm")

    # 联动临时方案
    size = (MARGIN + SIZE, MARGIN + HEAD + SIZE * 2)
    draw.multiline_text(size, "Fate 长 期 联 动", "black", font_cn, "md")
    size = (MARGIN + MIDDLE + SIZE * 7, MARGIN + HEAD + SIZE * 2)
    draw.multiline_text(size, "卫 星 预 告", "black", font_cn, "md")
    if not preview:
        size = (MARGIN + MIDDLE + SIZE * 7, MARGIN + HEAD + SIZE * 2.5)
        draw.multiline_text(size, "暂无", "black", font_cn, "mm")

    img = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    return img


def draw_circle(bg, pos):
    center = (pos[1] + RAW_SIZE // 2, pos[0] + RAW_SIZE // 2)
    cv2.circle(bg, center, (RAW_SIZE + 10) // 2, (77, 210, 255), 10)


def paste(bg, img_dir: pathlib.Path, positions: dict[str, list[list[int]]]):
    for name, pos in positions.items():
        im = read_profile(f"{img_dir / name}.png")
        for h, w in pos[1:]:
            _im = alpha_merge(im, bg[h, w])
            bg[h : h + _im.shape[0], w : w + _im.shape[1]] = _im
        del im
        if pos[0][0] == 0:
            draw_circle(bg, (pos[1][0], pos[1][1]))
    _p = positions["昔涟"][2]
    draw_circle(bg, (_p[0], _p[1]))
    return bg


def main():
    root = pathlib.Path("hsr")
    data, preview_data, special_data = load_history(root / "warps.csv")
    preview = False if len(preview_data) == 0 else True
    img_dir = root / "Icon"
    position, versions = get_position(data, preview_data, special_data)
    im = background("#f4f2ec", versions, preview)
    im = paste(im, img_dir, position)
    cv2.imwrite(str(root / "warps.jpg"), im)


main()
