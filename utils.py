"""
This module contains utility functions for standardizing images and evaluating generated images & depth-maps

"""

import json
import numpy as np
from typing import List
from pathlib import Path
from PIL import Image
from skimage.metrics import (
    normalized_root_mse as nrmse,
    structural_similarity as ssim,
    peak_signal_noise_ratio as psnr,
)
from rich.table import Table


# Local imports
from MANIQA.batch_predict import MANIQAScore


MANIQA_MODEL_FILEPATH = Path("./MANIQA/ckpt_kadid10k.pt")


def standardize_intensity(img: np.ndarray) -> np.ndarray:
    """
    Rescales intensity between 0-255

    """
    img = img - img.min()
    img = img / img.max() * 255

    return img.astype(np.uint8)


def calc_metrics(img1: Image, img2: Image) -> dict:
    """
    Calculates NRMSE, SSIM, and PSNR between two images

    """
    img1_np = np.asarray(img1.convert("L"))
    img2_np = np.asarray(img2.convert("L"))

    nrmse_val = nrmse(img1_np, img2_np)
    ssim_val = ssim(img1_np, img2_np)
    psnr_val = psnr(img1_np, img2_np)

    return {
        "NRMSE": nrmse_val,
        "SSIM": ssim_val,
        "PSNR": psnr_val,
    }


def maniqa_score(img_path: str) -> float:
    """
    Calculates the MANIQA score for an image

    """
    manr = MANIQAScore(ckpt_pth=MANIQA_MODEL_FILEPATH, cpu_num=8, num_crops=20)
    iqa_score = float(manr.predict_one(img_path).detach().cpu().numpy())

    return iqa_score


def create_table(
    base_dm: Image,
    calc_dms: List[Image],
    gen_img: Image,
    img_name: str,
    json_path: str | None = None,
) -> Table:
    """
    Create a table of metrics

    """
    table = Table(title="Quantitative Metrics", padding=(0, 0))
    table.add_column("Metric", justify="center", style="dark_magenta")
    table.add_column(img_name, justify="center", style="green")

    # Chosen metrics
    met_col = Table(show_edge=False)

    for i in ["NRMSE ↓", "SSIM ↑", "PSNR ↑"]:
        met_col.add_row(i)

    nrmses = [f"{calc_metrics(base_dm, dm)['NRMSE']:.3f}" for dm in calc_dms]
    ssims = [f"{calc_metrics(base_dm, dm)['SSIM']:.3f}" for dm in calc_dms]
    psnrs = [f"{calc_metrics(base_dm, dm)['PSNR']:.3f}" for dm in calc_dms]
    maniqa = [f"{maniqa_score(gen_img.filename):.3f}"]

    # Save to file
    res = {
        "NRMSE": nrmses,
        "SSIM": ssims,
        "PSNR": psnrs,
        "MANIQA": maniqa,
    }
    if json_path:
        with open(json_path, "w") as f:
            json.dump(res, f, indent=4)

    # Create table
    met_val = Table(show_edge=False)
    met_val.add_column("DepthAnything", justify="center")
    met_val.add_column("DepthAnything v2", justify="center")
    met_val.add_column("MiDaS", justify="center")
    met_val.add_row(*nrmses)
    met_val.add_row(*ssims)
    met_val.add_row(*psnrs)

    table.add_row(met_col, met_val)
    table.add_row("MANIQA ↑", *maniqa)

    return table
