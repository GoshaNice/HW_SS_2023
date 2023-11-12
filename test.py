import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

import src.model as module_model
from src.trainer import Trainer
from src.utils import ROOT_PATH
from src.utils.object_loading import get_dataloaders
from src.utils.parse_config import ConfigParser
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from torchmetrics.audio import PerceptualEvaluationSpeechQuality
import pyloudnorm as pyln
import torch.nn.functional as F
import numpy as np

def pad_to_target(prediction, target):
        if prediction.shape[-1] > target.shape[-1]:
            target = F.pad(
                target,
                (0, int(prediction.shape[-1] - target.shape[-1])),
                "constant",
                0,
            )
        elif prediction.shape[-1] < target.shape[-1]:
            prediction = F.pad(
                prediction,
                (0, int(target.shape[-1] - prediction.shape[-1])),
                "constant",
                0,
            )
        return prediction, target


DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def main(config, out_file):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"We are running on {device}")

    # setup data_loader instances
    dataloaders = get_dataloaders(config)

    # build model architecture
    model = config.init_obj(config["arch"], module_model)
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    calc_sisdr = ScaleInvariantSignalDistortionRatio()
    calc_pesq = PerceptualEvaluationSpeechQuality(16000, "wb")
    results = []
    si_sdrs = []
    pesqs = []


    with torch.no_grad():
        for batch_num, batch in enumerate(tqdm(dataloaders["test"])):
            batch = Trainer.move_batch_to_device(batch, device)
            s1, s2, s3, probs = model(**batch)
            batch["prediction"] = s1

            for i in range(len(batch["prediction"])):
                prediction = batch["prediction"][i]
                target = batch["target"][i].unsqueeze(0)
                prediction, target = pad_to_target(prediction, target)
                prediction = prediction.squeeze(0).detach().cpu().numpy()
                target = target.squeeze(0).detach().cpu().numpy()

                meter = pyln.Meter(16000) # create BS.1770 meter
                loud_prediction = meter.integrated_loudness(prediction)
                loud_target = meter.integrated_loudness(target)

                prediction = pyln.normalize.loudness(prediction, loud_prediction, -20)
                target = pyln.normalize.loudness(target, loud_target, -20)

                si_sdr = calc_sisdr(torch.from_numpy(prediction), torch.from_numpy(target))
                pesq = calc_pesq(torch.from_numpy(prediction), torch.from_numpy(target))

                si_sdrs.append(si_sdr.item())
                pesqs.append(pesq.item())

                results.append(
                    {
                        "SI-SDR": si_sdr.item(),
                        "PESQ": pesq.item()
                    }
                )

    print("Final_metrics")
    print("SI-SDR: ", sum(si_sdrs) / len(si_sdrs))
    print("PESQ: ", sum(pesqs) / len(pesqs))

    with Path(out_file).open("w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        default="output.json",
        type=str,
        help="File to write results (.json)",
    )
    args.add_argument(
        "-t",
        "--test-data-folder",
        default=None,
        type=str,
        help="Path to dataset",
    )
    args.add_argument(
        "-b",
        "--batch-size",
        default=20,
        type=int,
        help="Test dataset batch size",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    # if `--test-data-folder` was provided, set it as a default test set
    if args.test_data_folder is not None:
        test_data_folder = Path(args.test_data_folder).absolute().resolve()
        assert test_data_folder.exists()
        config.config["data"] = {
            "test": {
                "batch_size": args.batch_size,
                "num_workers": args.jobs,
                "datasets": [
                    {
                        "type": "CustomDirAudioDataset",
                        "args": {
                            "mix_dir": str(test_data_folder / "mix"),
                            "refs_dir": str(
                                test_data_folder / "refs"
                            ),
                            "targets_dir": str(
                                test_data_folder / "targets"
                            )
                        },
                    }
                ],
            }
        }

    assert config.config.get("data", {}).get("test", None) is not None
    config["data"]["test"]["batch_size"] = args.batch_size
    config["data"]["test"]["n_jobs"] = args.jobs

    main(config, args.output)
