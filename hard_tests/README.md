# How to use `evaluate_hard_tests.py`

This `evaluate_hard_tests.py` script is designed to perform a complete workflow from inference on an image dataset to computing and ranking evaluation metrics for the S3OD background removal model.

## Purpose

- Load a pre-trained S3OD model.
- Perform inference for background removal on a set of input images.
- Save the predicted masks and output RGBA images.
- Compute evaluation metrics (MAE, MaxF, AvgF, Sm, Em, wF) for the quality of background removal against ground truth.
- Store and rank these metrics against previous runs for easy comparison.

## Prerequisites

Before running the script, ensure you have the necessary libraries installed and the file/folder structure is set up correctly:

1.  **Clone the S3OD repository:**
    ```bash
    !git clone https://github.com/AmnO-O/S3OD.git
    %cd S3OD
    ```

2.  **Install required packages:**
    ```bash
    !pip -q install -e .
    !pip -q install hydra-core albumentations opencv-python scipy tqdm
    ```

3.  **Download the DIS5K dataset:** Ensure the `DIS5K` dataset is downloaded and extracted into the `datasets/DIS5K/DIS5K` folder within the root of your S3OD repository. The directory structure should look like this:
    ```
    S3OD/
    ├── datasets/
    │   └── DIS5K/
    │       └── DIS5K/
    │           ├── DIS-TE1/
    │           │   ├── gt/  (contains ground truth masks)
    │           │   └── im/  (contains input images)
    │           ├── DIS-TE2/
    │           └── ...
    └── ...
    ```
    You can download and extract it using the following commands:
    ```bash
    !gdown --id 1O1eIuXX1hlGsV7qx4eSkjH231q7G1by1 -O DIS5K.zip
    !mkdir -p datasets/DIS5K
    !unzip -q DIS5K.zip -d datasets/DIS5K
    !rm DIS5K.zip
    ```

4.  **Download the model checkpoint:** Ensure the checkpoint file (`last_focal_iou_ssim.ckpt`) is in the root directory of your S3OD repository.
    ```bash
    !wget -O last_focal_iou_ssim.ckpt \
      "https://huggingface.co/AmnO-O/S3OD-DIS5K-Finetune/resolve/main/focal_iou_ssim/last.ckpt"
    ```

## How to Use

After setting up the prerequisites, you can run the script from your terminal in the root directory of `S3OD`:

```bash
python evaluate_hard_tests.py
```

The script will automatically:

1.  Initialize the `BackgroundRemoval` model and load the downloaded checkpoint.
2.  Perform inference on images located in `datasets/DIS5K/DIS5K/DIS-TE1/im`.
3.  Save the predicted masks and RGBA images to `outputs/DIS-TE1/masks` and `outputs/DIS-TE1/rgba` respectively.
4.  Compute evaluation metrics by comparing the inference results with the `ground truth` in `datasets/DIS5K/DIS5K/DIS-TE1/gt`.
5.  Save the metrics to `s3od_metrics_results.csv` and print a ranked table of the metrics.

## Configuration

The main configuration paths and parameters are defined within the `if __name__ == '__main__':` block of the script:

-   `repo_root`: Path to the root directory of the S3OD repository. Defaults to `.` (current directory).
-   `ckpt_path`: Path to the model checkpoint file. Defaults to `last_focal_iou_ssim.ckpt`.
-   `images_dir`: Directory containing the input images for inference.
-   `gt_dir`: Directory containing the ground truth masks for evaluation.
-   `output_infer_dir`: Main output directory for inference results.
-   `pred_dir`: Specific directory containing the predicted masks used for metric computation. (`output_infer_dir + "/masks"`)
-   `device`: Device to run the model on (`"cuda"` if GPU is available, `"cpu"` otherwise).
-   `run_name`: A string identifier for the current run, used for logging to the CSV results file.

You can modify these variables directly within the `evaluate_hard_tests.py` script to suit your configuration.

## Output Results

The script will create the following directories and files:

-   `outputs/DIS-TE1/masks/`: Contains the predicted binary masks (grayscale images) for each input image.
-   `outputs/DIS-TE1/rgba/`: Contains the RGBA images (original images with background removed) for each input image.
-   `s3od_metrics_results.csv`: A CSV file containing the evaluation metrics of the current and previous runs, along with a ranked table.

In the console, you will see inference progress followed by the ranked evaluation metrics table.
