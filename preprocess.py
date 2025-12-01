import argparse
import os
import numpy as np
from scipy.io import wavfile
from audiomentations import Compose, AddGaussianSNR, TimeStretch, PitchShift, AddBackgroundNoise, ApplyImpulseResponse
from audiomentations.core.audio_loading_utils import load_sound_file
import nlpaug.augmenter.audio as naa
import nlpaug.flow as naf
import pandas as pd
from tqdm import tqdm
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
warnings.filterwarnings("ignore", category=UserWarning)

BASE = "./data/train/train"
background_noises_path = os.path.join(BASE, "background_noises")
rir_path = os.path.join(BASE, "rir")

AUG1 = None
AUG2 = None
SR = 16000

def init_worker(aug_type):
    """
    Called ONCE per worker process.
    Creates augmentation pipelines once → avoids huge overhead.
    """
    global AUG1, AUG2, SR
    AUG1, AUG2 = get_augmenters(aug_type)
    print(f"[Worker Init] Augmenters initialized for aug_type={aug_type}")


# -------------------------------
# Build augmentation pipelines
# -------------------------------
def get_augmenters(aug_type):
    aug1 = naf.Sometimes([
        naa.VtlpAug(sampling_rate=SR, zone=(0.0, 1.0), coverage=1.0, factor=(0.9, 1.1)),
    ], aug_p=0.4)

    aug_TPG = Compose([
        AddGaussianSNR(min_snr_db=10, max_snr_db=30, p=0.2),
        TimeStretch(min_rate=0.8, max_rate=1.2, leave_length_unchanged=False, p=0.4),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.4),
    ])

    aug_TPGB = Compose([
        AddGaussianSNR(min_snr_db=10, max_snr_db=30, p=0.2),
        TimeStretch(min_rate=0.8, max_rate=1.2, leave_length_unchanged=False, p=0.4),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.4),
        AddBackgroundNoise(
            sounds_path=background_noises_path,
            min_snr_db=10,
            max_snr_db=30,
            p=0.4),
    ])

    aug_TPGBIR = Compose([
        AddGaussianSNR(min_snr_db=10, max_snr_db=30, p=0.2),
        TimeStretch(min_rate=0.8, max_rate=1.2, leave_length_unchanged=False, p=0.4),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.4),
        AddBackgroundNoise(
            sounds_path=background_noises_path,
            min_snr_db=10,
            max_snr_db=30,
            p=0.4),
        ApplyImpulseResponse(ir_path=rir_path, p=0.4)
    ])

    return aug1, {"TPG": aug_TPG, "TPGB": aug_TPGB, "TPGBIR": aug_TPGBIR}[aug_type]

def process_single_file(file, inLabels_dict, InPath, OutPath, n_augmentations):
    """
    Runs inside each process.
    Uses global AUG1 and AUG2 that were set in init_worker().
    """
    global AUG1, AUG2

    try:
        file_id = os.path.splitext(file)[0].lower().strip()
        text = inLabels_dict.get(file_id, None)

        if text is None:
            return {"error": f"No label for {file}", "labels": []}

        # Load WAV
        samples, sample_rate = load_sound_file(
            os.path.join(InPath, file),
            sample_rate=None
        )

        results = []

        for i in range(n_augmentations):
            outname = f"{file_id}_aug{i+1}.wav"

            # Aug 1
            aug1 = AUG1.augment(samples)
            src = aug1[0] if isinstance(aug1, (list, tuple)) else aug1

            # Aug 2
            aug2 = AUG2(samples=src, sample_rate=sample_rate)

            # Save
            wavfile.write(
                os.path.join(OutPath, outname),
                rate=sample_rate,
                data=aug2
            )

            results.append({"file_name": outname, "transcription": text})

        return {"error": None, "labels": results}

    except Exception as e:
        return {"error": f"Error processing {file}: {e}", "labels": []}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aug_type", type=str, required=True)
    parser.add_argument("--src_dir", type=str, required=True)
    parser.add_argument("--label_file", type=str, required=True)
    parser.add_argument("--dist_dir", type=str, required=True)
    parser.add_argument("--n_augmentations", type=int, required=True)
    parser.add_argument("--n_workers", type=int, default=None)
    args = parser.parse_args()

    if args.n_workers is None:
        args.n_workers = multiprocessing.cpu_count()

    InPath = os.path.join(BASE, args.src_dir)
    InLabelPath = os.path.join(BASE, args.label_file)
    OutPath = os.path.join(BASE, args.dist_dir)
    OutLabelPath = os.path.join(OutPath, "metadata.csv")

    os.system(f"rm -rf {OutPath}")
    os.makedirs(OutPath, exist_ok=True)

    # Load label CSV once
    df = pd.read_csv(InLabelPath)
    df["id_norm"] = df["id"].str.lower().str.strip()
    inLabels_dict = dict(zip(df["id_norm"], df["text"]))

    files = [f for f in os.listdir(InPath) if f.endswith(".wav")]
    print(f"Found {len(files)} audio files")

    outLabels = []
    errors = []

    # Multiprocessing with initializer (important!)
    with ProcessPoolExecutor(
        max_workers=args.n_workers,
        initializer=init_worker,
        initargs=(args.aug_type,)
    ) as executor:

        futures = {
            executor.submit(
                process_single_file,
                file,
                inLabels_dict,
                InPath,
                OutPath,
                args.n_augmentations
            ): file
            for file in files
        }

        with tqdm(total=len(files), desc="Processing", unit="file") as pbar:
            for fut in as_completed(futures):
                file = futures[fut]
                try:
                    result = fut.result()
                    if result["error"]:
                        errors.append(result["error"])
                    else:
                        outLabels.extend(result["labels"])
                except Exception as e:
                    errors.append(f"Crash on {file}: {e}")
                pbar.update(1)

    if outLabels:
        pd.DataFrame(outLabels).to_csv(OutLabelPath, index=False)
        print(f"✓ Saved {len(outLabels)} augmented samples → {OutLabelPath}")

    if errors:
        with open(os.path.join(OutPath, "errors.log"), "w") as f:
            f.write("\n".join(errors))
        print(f"⚠ {len(errors)} errors → see errors.log")


if __name__ == "__main__":
    main()
