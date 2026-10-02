# Dataset

This project uses the **RSNA Pneumonia Detection Challenge** dataset from Kaggle.
The data is **not included** in this repository (large size and Kaggle's terms).

## How to get it

1. Create a free Kaggle account: https://www.kaggle.com
2. Open the competition page and accept the rules:
   https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
3. Download the data, for example from Colab with `kagglehub`:

   ```python
   import kagglehub
   kagglehub.login()   # paste your Kaggle API token when asked
   path = kagglehub.competition_download("rsna-pneumonia-detection-challenge")
   print(path)
   ```

4. The downloaded folder should contain:

   ```
   stage_2_train_labels.csv
   stage_2_detailed_class_info.csv
   stage_2_train_images/      (DICOM .dcm files)
   stage_2_test_images/
   stage_2_sample_submission.csv
   ```

5. In the notebook, set the `path` variable to this folder.

## Labels

`stage_2_train_labels.csv` has one or more rows per patient:

| Column | Meaning |
|---|---|
| patientId | X-ray ID (matches the `.dcm` filename) |
| x, y, width, height | Pneumonia bounding box (empty for normal X-rays) |
| Target | 1 = pneumonia, 0 = no pneumonia |

Patients with more than one pneumonia region have several rows. Bounding boxes
are used only to evaluate localization, never for training.

## Sample images

A few example PNGs used for the demo are in `../demo_samples/`.
