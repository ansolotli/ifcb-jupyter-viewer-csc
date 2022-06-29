from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def extract_sample_images(sample, raw_dir, out_dir, exist_ok=False):
    try:
        adc = next(Path(raw_dir).glob(f"**/{sample}.adc"))
    except StopIteration:
        print(f"Sample {sample} not found in {raw_dir}")
        raise
    roi = adc.with_suffix(".roi")
    raw_to_png(adc, roi, out_dir, force=exist_ok)


def raw_to_png(adc, roi, out_dir=None, force=False):
    adc = Path(adc)
    roi = Path(roi)
    for f in (adc, roi):
        if not f.is_file():
            raise FileNotFoundError(f)
    sample = adc.with_suffix("").name
    out_dir = Path(adc.with_suffix("")) if not out_dir else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=force)
    # Read bytes from .roi-file into 8-bit integers
    roi_data = np.fromfile(roi, dtype="uint8")
    # Parse each line of .adc-file
    with open(adc) as adc_fh:
        for i, line in enumerate(adc_fh, start=1):
            line = line.split(",")
            roi_x = int(line[15])  # ROI width
            roi_y = int(line[16])  # ROI height
            start = int(line[17])  # start byte
            # Skip empty roi
            if roi_x < 1 or roi_y < 1:
                continue
            # roi_data is a 1-dimensional array, where
            # all roi are stacked one after another.
            end = start + (roi_x * roi_y)
            # Reshape into 2-dimensions
            img = roi_data[start:end].reshape((roi_y, roi_x))
            img_path = out_dir / f"{sample}_{i:05}.png"
            # imwrite reshapes automatically to 3-dimensions (RGB)
            Image.fromarray(img).save(img_path)


def prediction_dataframe(probabilities, thresholds=0.0):
    if isinstance(probabilities, list):
        # Need to join multiple csv-files as one df
        df_list = []
        for csv in probabilities:
            df = pd.read_csv(csv)
            # Create multi-index from sample name and roi number
            df.insert(0, "sample", Path(csv).with_suffix("").stem)
            df.set_index(["sample", "roi"], inplace=True)
            df_list.append(df)
        df = pd.concat(df_list)
    elif isinstance(probabilities, (str, Path)):
        df = pd.read_csv(probabilities, index_col=0)
    else:
        raise ValueError(f"Type {type(probabilities)} not allowed for probabilities")
    if isinstance(thresholds, (str, Path)):
        thresholds = threshold_dictionary(thresholds)
    # Insert 'prediction' and 'classified' columns to dataframe
    if not df.empty:
        insert_prediction(df, thresholds)
    return df


def threshold_dictionary(thresholds, default=None):
    thres_dict = {}
    with open(thresholds) as fh:
        for line in fh:
            line = line.strip().split()
            key = line[0]
            if len(line) > 1:
                value = float(line[1])
            elif default:
                value = float(default)
            else:
                raise ValueError(
                    f"Missing threshold for {key}, and no default value specified."
                )
            thres_dict[key] = value
    return thres_dict


def row_prediction(row, thresholds):
    if isinstance(thresholds, (int, float)):
        name = row.idxmax()
        return (name, row[name] > thresholds)
    # Generator for all classes that have condidence above
    # their specific threshold (sorted by probability, descending)
    above_threshold = (
        (name, True)
        for name, probability in row.sort_values(ascending=False).items()
        if name in thresholds and probability >= thresholds[name]
    )
    try:
        # Return the class with the highest probability above a threshold.
        # I.e. a class that is classified
        return next(above_threshold)
    except StopIteration:
        # If none are classified, return the class with highest overall probability
        return (row.idxmax(), False)


def insert_prediction(df, thresholds):
    """This function modifies `df` in place"""
    preds, status = zip(*df.apply(row_prediction, axis=1, args=(thresholds,)))
    df.insert(0, "prediction", preds)
    df["prediction"] = df["prediction"].astype("category")
    df.insert(1, "classified", status)
