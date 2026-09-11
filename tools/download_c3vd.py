"""C3VD 数据集下载脚本（Colonoscopy 3D Video Dataset）。

C3VD 官方数据托管在 Google Drive（CC BY-NC-SA 4.0，非商用许可）。
共 22 个注册视频（约 90GB）+ 4 个筛查视频，每帧含配对深度/法线/光流/遮挡/位姿。

用法（需能访问 Google Drive 的环境，或已配置代理）:

    # 下载单个序列（ColonCrafter 官方示例，2.86GB）
    python tools/download_c3vd.py --root data/c3vd cecum_t1_a

    # 下载多个序列
    python tools/download_c3vd.py --root data/c3vd cecum_t1_a cecum_t1_b

    # 下载全部注册视频（约 90GB）
    python tools/download_c3vd.py --root data/c3vd --all

下载后每个序列解压为: <root>/<seq>/color/*.png  depth/*.tiff  pose.txt
"""

import argparse
import os
import subprocess
import sys
import zipfile

# 22 个注册视频 + 4 个筛查视频的 Google Drive file_id
REGISTERED = {
    "cecum_t1_a": "14o6_4GQLZWx5dQq2L_drzmN_rlCT7Yhr",
    "cecum_t1_b": "1z3AHdnBH_YoCMnnTfDa8SNPQYIsvaBO3",
    "cecum_t2_a": "13XhJIev9memFtwUf_dnjJ7o8z6O_c-xW",
    "cecum_t2_b": "1ykYtQGiFesev5QLfz_avYuQ5a7Zs8kgF",
    "cecum_t2_c": "1tNoBLpPbrQexKlnOKMK2peERn9Rj_9Dp",
    "cecum_t3_a": "1Uw8uCRRDm_RrgkccGbiBXZHf9P-THM2Q",
    "cecum_t4_a": "1FC-dR__0LVb7WH02KpUx9TZVvvvN-Gyx",
    "cecum_t4_b": "11SbH2AZsuciTu3iGxdXCdQZky6uDTyS5",
    "desc_t4_a": "1d9HDNg4-Og1cTWM-eIU5SWM2BMrMWhwQ",
    "sigmoid_t1_a": "19VGDuZ73OWNwM8eIgDkkZYBPJPQ5BD91",
    "sigmoid_t2_a": "16epAys428g9vBQgm611TElMyAXORo7rH",
    "sigmoid_t3_a": "1ZRU2KuHoc2XCbKSY_A1S-7BxEfKf9xPr",
    "sigmoid_t3_b": "1XfZFAQ5_Wxle8d5wSlOCumKSg4IP8wTv",
    "trans_t1_a": "1urFuVo8ZalwPmsXEZg3xzhuqhpgWV8hw",
    "trans_t1_b": "1hyjmd7vn86McE1nUnYCzvOm8LlyHLYwt",
    "trans_t2_a": "1ylZWWtVlXfDx9dhPIeWJ1HqDHJZ3QKdH",
    "trans_t2_b": "1vru228_TEgxT3aS90CmvOWsMB0RLnAxn",
    "trans_t2_c": "12YpowbP6zhoO_Qx9UBwhfRLXNJN1EAu4",
    "trans_t3_a": "1B4aeZfAqmUJgWr8e-2YAibUe4er30ncr",
    "trans_t3_b": "1ZpbYcDVP-sCTjjQrDc303olFgsr2nA5J",
    "trans_t4_a": "18qzXMifS54jAx29yROKXXxxZg0qo-iTz",
    "trans_t4_b": "1C-nw6MR7sxssw3LS-GpiPmwBzEYhUCHN",
}

SCREENING = {
    "screening_t1": "18m3Z5zJtljor_AGmPW8OgO9fRuactuNk",
    "screening_t2": "1kn_qevX7lLh3gkiKAt3hgWFpy0P68s6",
    "screening_t3": "1RmOnnjJBzCMwO5gPY4h3e4MpcDDOvOb6",
    "screening_t4": "1sYps79WjJ0ETRtuWtHd_1zeuyPLtoMM7",
}

ALL = {**REGISTERED, **SCREENING}


def _gdown(fid, out_path):
    try:
        import gdown
    except ImportError:
        print("需要 gdown: pip install gdown")
        sys.exit(1)
    gdown.download(id=fid, output=out_path, quiet=False)


def _unzip(zip_path, root, name):
    dest = os.path.join(root, name)
    os.makedirs(dest, exist_ok=True)
    print("解压 %s -> %s" % (zip_path, dest))
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    os.remove(zip_path)


def main():
    parser = argparse.ArgumentParser(description="下载 C3VD 数据集（需能访问 Google Drive）")
    parser.add_argument("sequences", nargs="*", help="要下载的序列名（见脚本内 REGISTERED/SCREENING）")
    parser.add_argument("--root", default="data/c3vd", help="数据根目录")
    parser.add_argument("--all", action="store_true", help="下载全部 26 个序列（约 120GB）")
    parser.add_argument("--no-unzip", action="store_true", help="只下载 zip 不解压")
    args = parser.parse_args()

    if args.all:
        names = list(ALL)
    elif args.sequences:
        names = args.sequences
    else:
        print("可用序列（REGISTERED 22 个）:")
        for k in REGISTERED:
            print("  %s" % k)
        print("筛查视频（SCREENING 4 个）:")
        for k in SCREENING:
            print("  %s" % k)
        print("\n用法: python tools/download_c3vd.py --root data/c3vd <序列名> [更多序列名...]")
        sys.exit(0)

    os.makedirs(args.root, exist_ok=True)
    for name in names:
        if name not in ALL:
            print("未知序列 %r，跳过（可用 --all 或查看脚本内清单）" % name)
            continue
        fid = ALL[name]
        zip_path = os.path.join(args.root, name + ".zip")
        print("下载 %s (id=%s)" % (name, fid))
        _gdown(fid, zip_path)
        if not args.no_unzip:
            _unzip(zip_path, args.root, name)
        print("完成 %s" % name)


if __name__ == "__main__":
    main()
