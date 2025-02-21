# DeepChecker

A fast and accurate image duplication detection tool that combines CLIP (Contrastive Language-Image Pre-Training) and SIFT (Scale-Invariant Feature Transform) algorithms to identify duplicate and similar images.
<img src="https://github.com/user-attachments/assets/f3367190-93fc-4e6e-9b26-b0ed1f31a2e2" alt="image" style="zoom:35%;" />

## Features

- **Dual-Algorithm Analysis**: Combines CLIP for semantic similarity and SIFT for local feature matching
- **Multiple Format Support**: Handles various image formats (JPEG, PNG, GIF, BMP, TIFF) and PDFs
- **Real-time Progress**: Shows analysis progress during SIFT verification
- **Interactive UI**: Drag-and-drop interface with file preview
- **Detailed Results**:
  - Top 5 similar pairs with similarity scores
  - Duplicate image groups
  - Local feature matches count
  - CLIP similarity scores
- **Export Functionality**: Download top 20 similar pairs as CSV



## Tech Stack

### Frontend

- React
- TypeScript
- Tailwind CSS
- Lucide Icons

### Backend

- Python
- FastAPI
- OpenCV (SIFT)
- CLIP (OpenAI)
- PyTorch



# Use it on server

1. **Upload Images**:
   - Drag and drop images/PDFs into the upload area
   - Or click "Select Files" to choose files
   - Or click "Select Folder" to upload an entire folder

2. **Analysis**:
   - Click "Analyze" to start processing
   - Watch the progress bar for SIFT verification progress
   - View results in real-time

3. **Results**:
   - View top 5 similar pairs with similarity scores
   - Check duplicate groups if found
   - Download top 20 similar pairs as CSV

4. **CSV Export Format**:
   - Pair Number
   - Image 1 filename
   - Image 2 filename
   - CLIP Score (%)
   - Local Matches count



# Deploy server

```
# front end 
mamba install nodejs
```



```
conda activate dev

# back
python app.py

# front
npm run dev
```





# Deploy local by conda 

download conda:
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh 
```



Setup:

```
mamba create -n dev python=3.12 -y 
conda activate dev 
mamba install numpy pillow opencv pytorch torchvision openai-clip glob2 pdf2image ipykernel -y 
# mamba install ipykernel -y 
# mamba install openai-clip glob2 -y
# pip install ftfy regex tqdm 
```



run script:

```

```





# Change other models

[sentence-transformers/clip-ViT-B-32 · Hugging Face](https://huggingface.co/sentence-transformers/clip-ViT-B-32)

|                            Model                             | Top 1 Performance |
| :----------------------------------------------------------: | ----------------- |
| [clip-ViT-B-32](https://huggingface.co/sentence-transformers/clip-ViT-B-32) | 63.3              |
| [clip-ViT-B-16](https://huggingface.co/sentence-transformers/clip-ViT-B-16) | 68.1              |
| [clip-ViT-L-14](https://huggingface.co/sentence-transformers/clip-ViT-L-14) | 75.4              |

For a multilingual version of the CLIP model for 50+ languages have a look at: [clip-ViT-B-32-multilingual-v1](https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1)



# Others

## Limitations

- Maximum recommended batch size: Depends on available memory
- PDF processing may take longer due to conversion
- CLIP analysis requires GPU for optimal performance

## Credits

Developed by UM_DengLab

- Contact: Peng, yc47680@um.edu.mo
- GitHub: https://github.com/mugpeng
