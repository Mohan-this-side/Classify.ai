# 🔐 Kaggle API Setup Instructions

## Step 1: Get Your Kaggle API Token

1. Go to [Kaggle.com](https://www.kaggle.com) and log in
2. Click on your profile picture (top right corner)
3. Select "Account" from the dropdown
4. Scroll down to the "API" section
5. Click "Create New API Token"
6. This will download a file called `kaggle.json`

## Step 2: Place the API Token

1. Move the downloaded `kaggle.json` file to: `~/.kaggle/kaggle.json`
2. The file should contain:
```json
{
  "username": "your_username",
  "key": "your_api_key"
}
```

## Step 3: Set Permissions

Run this command in your terminal:
```bash
chmod 600 ~/.kaggle/kaggle.json
```

## Step 4: Test the Setup

Once you've completed the above steps, run:
```bash
kaggle datasets list
```

If it works, you should see a list of datasets.

## Step 5: Download the Spotify Dataset

After authentication is working, run:
```bash
cd test_data
kaggle datasets download nabihazahid/spotify-dataset-for-churn-analysis
unzip spotify-dataset-for-churn-analysis.zip
```

## Alternative: Manual Download

If you prefer not to use the API, you can:
1. Go to the dataset page: https://www.kaggle.com/datasets/nabihazahid/spotify-dataset-for-churn-analysis
2. Click "Download" button
3. Extract the files to the `test_data` folder
