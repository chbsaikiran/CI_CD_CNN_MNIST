## Create a virtual environment and activate it:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

## Install dependencies:
pip install -r requirements.txt

## Train the model:
python train.py

## Run tests:
pytest tests/

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