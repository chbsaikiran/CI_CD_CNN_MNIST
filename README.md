[![ML Pipeline](https://github.com/chbsaikiran/CI_CD_CNN_MNIST/actions/workflows/ml-pipeline.yml/badge.svg)](https://github.com/chbsaikiran/CI_CD_CNN_MNIST/actions/workflows/ml-pipeline.yml)

## Create a virtual environment and activate it:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

## Install dependencies:
pip install -r requirements.txt

## Train the model:
python train.py

## Run tests:
pytest -v --capture=no tests/

## To push to GitHub:
Create a new repository on GitHub
Initialize git and push:
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main


## The GitHub Actions workflow will automatically trigger when you push to the repository. It will:
1. Set up a Python environment
2. Install dependencies
3. Train the model
4. Run all tests
5. Save the model artifacts

## The tests check for:
1. Model parameter count (< 25000)
2. Input shape compatibility (28x28)
3. Output shape (10 classes)
4. Model accuracy (> 95%)



## Three additional tests after using augmented data
1. test_combined_transformations:
This test evaluates the model's accuracy when subjected to combined transformations (shear, rotation, and scale). It creates a dataset with equal proportions of images transformed by these techniques, checks the model's performance, and ensures that the accuracy exceeds 90%. It also visualizes the transformations for analysis.

2. test_data_integrity:
This test ensures that the transformations (shear, rotation, scale) preserve the structural integrity of the images. It validates image dimensions, pixel value ranges, and similarity (SSIM) between the original and transformed images to ensure the augmentations do not distort data beyond acceptable limits.

3. test_augmentation_diversity:
This test checks if the augmentations produce diverse variations of images. It measures pixel variance across multiple versions of augmented images and structural similarity (SSIM) between them, ensuring that each transformation introduces sufficient variety without being overly repetitive. It also provides a visual comparison of the diversity.