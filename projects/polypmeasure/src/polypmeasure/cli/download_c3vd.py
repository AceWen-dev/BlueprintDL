"""Download selected C3VD archives for PolypMeasure research workflows."""

import argparse
import os
import zipfile


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

ALL_SEQUENCES = {**REGISTERED, **SCREENING}


def _download(file_id, output_path):
    try:
        import gdown
    except ImportError as exc:
        raise SystemExit(
            "缺少可选依赖 gdown；请在 PolypMeasure 环境执行 "
            "uv sync --extra data"
        ) from exc
    gdown.download(id=file_id, output=output_path, quiet=False)


def _unzip(zip_path, root, name):
    destination = os.path.join(root, name)
    os.makedirs(destination, exist_ok=True)
    print("解压 %s -> %s" % (zip_path, destination))
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(destination)
    os.remove(zip_path)


def build_parser():
    parser = argparse.ArgumentParser(description="下载 C3VD 数据集")
    parser.add_argument("sequences", nargs="*")
    parser.add_argument("--root", default="data/c3vd")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--no-unzip", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.all:
        names = list(ALL_SEQUENCES)
    elif args.sequences:
        names = args.sequences
    else:
        print("可用注册视频:", ", ".join(REGISTERED))
        print("可用筛查视频:", ", ".join(SCREENING))
        return

    os.makedirs(args.root, exist_ok=True)
    for name in names:
        if name not in ALL_SEQUENCES:
            print("未知序列 %r，跳过" % name)
            continue
        archive_path = os.path.join(args.root, name + ".zip")
        print("下载 %s" % name)
        _download(ALL_SEQUENCES[name], archive_path)
        if not args.no_unzip:
            _unzip(archive_path, args.root, name)
        print("完成 %s" % name)


if __name__ == "__main__":
    main()
