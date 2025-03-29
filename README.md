# Captcha by TKL

A simple project that processes a given Captcha image and produces the results leverage a CNN classifier and OCR via Tesseract.

The project leverages two established approaches to classify the characters within the Captcha images. The first approach is a Convolutional Neural Network (CNN) model that was trained on the given samples. The second approach is to leverage Tesseract, an OCR engine, to extract the text from the Captcha images.

## Premise 🌟
- The number of characters remains the same for all captchas
- The font and spacing is the same for all captchas
- Background and foreground colors and texture remain largely the same
- There is no skew or rotation in the structure of the characters
- The captcha generator creates strictly 5-character captchas, and each character is either an upper-case character (A-Z) or a numeral (0-9)
- A total of 25 training samples were provided as training sample
- Each image has a fixed size of 60x30 pixels

## Installation 📥

```bash
pip install git+https://github.com/tkl7665/Captcha.git
```

## Usage

### Command Line
```bash
captcha input100.jpg result100.txt
```
#### Example
```bash
captcha ..\input\input100.jpg ../output/result100.txt
captcha.app INFO Processing ..\input\input100.jpg
captcha.app INFO Initializing CNN model...
captcha.app INFO loading from C:\_tkl_mock\02_Work\99_Test\t03\venv\lib\site-packages\captcha\models\classIndex.json
captcha.app INFO loading from C:\_tkl_mock\02_Work\99_Test\t03\venv\lib\site-packages\captcha\models\cnnModel.pth
captcha.app INFO Loaded CNN model with {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29, 'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35}
captcha.app INFO Using CNN result as Tesseract False
captcha.app INFO Final Result: YMB1Q
captcha.configs.cleanup INFO Starting cleanup
captcha.configs.cleanup INFO Removing ./captcha/output//cd9a_1_cd9a_input100.jpg
captcha.configs.cleanup INFO Removing ./captcha/output//cd9a_4_cd9a_input100.jpg
captcha.configs.cleanup INFO Removing ./captcha/output//cd9a_2_cd9a_input100.jpg
captcha.configs.cleanup INFO Removing ./captcha/output//cd9a_input100.jpg
captcha.configs.cleanup INFO Removing ./captcha/output//cd9a_0_cd9a_input100.jpg
captcha.configs.cleanup INFO Removing ./captcha/output//cd9a_3_cd9a_input100.jpg
```

### Python
```python
from captcha.app import interactiveMode
interactiveMode('../output/')
```

#### Example
```python
>>> from captcha.app import interactiveMode
>>> interactiveMode('../output/')
captcha.app INFO Initializing...
captcha.app INFO Checking Tesseract...
captcha.ocr WARNING Tesseract not found: tesseract is not installed or it is not in your PATH. See README file for more information.
captcha.app INFO Tesseract: False
captcha.app INFO Default: CNN
captcha.app INFO Initializing CNN model...
captcha.app INFO loading from C:\_tkl_mock\02_Work\06_Captcha\01_Dev\captcha\models\classIndex.json
captcha.app INFO loading from C:\_tkl_mock\02_Work\06_Captcha\01_Dev\captcha\models\cnnModel.pth
captcha.app INFO Loaded CNN model with {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29, 'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35}
captcha.app INFO Saving Predictions at: C:/_tkl_mock/02_Work/06_Captcha/output/6bf3
Enter image path (0 to quit <CNN>): ../input/input100.jpg
captcha.app INFO Using CNN result as Tesseract False
captcha.app INFO Final Result: YMB1Q
captcha.app INFO YMB1Q written to ../output//6bf3/prediction_1.txt
Enter image path (0 to quit <CNN>):
```

## CNN Model 🧠
A Convolutional Neural Network (CNN) model was trained on the given samples to classify the characters within the Captcha images. The model commited within the code was trained without a GPU on augmented data due to the limited number of samples provided.

Read more about CNN [here](https://en.wikipedia.org/wiki/Convolutional_neural_network) and [here](https://www.datacamp.com/tutorial/introduction-to-convolutional-neural-networks-cnns).

### Training
The latest trained model would be saved in the folder specified while a copy would also be created as a sub-folder for tracking purposes. This is to ensure all versions of trained models are kept for tracking purposes. Future work would be to integrate this with MLOps solution.

trainCNN.py
```python
main(<training data folder>,<folder to save model>)
```

The CNN model was designed with a simple structure and trained for 10 epochs with learning rate of 0.001:

Input Layer (Grayscale Image)  
│  
├─ Conv2D Layer: 1 input channel → 32 output channels (3×3 kernel, padding=1)  
│  │  
│  └─ ReLU Activation → MaxPool (2×2)  
│  
├─ Conv2D Layer: 32 → 64 channels (3×3 kernel, padding=1)  
│  │  
│  └─ ReLU Activation → MaxPool (2×2)  
│  
├─ Flatten → Fully Connected Layer (64×2×2 → 128 units)  
│  │  
│  └─ ReLU Activation  
│  
└─ Output Layer (128 → numClasses units)

### Data 📊
A total of twenty-five samples was provided where each sample was a 5-character captcha image. The images were can be found within the samples/input folder.

#### Preprocessing 🔧
Code is kept in preprocessing.py
```python
cropAllImages(<input folder>,<output folder>,<label folder>)
```
- Crop image to remove white space surrounding the characters
- Split the image into 5 individual characters
- Below is the distribution of the number of training samples for each character based on the twenty five input samples. A total of 125 samples where spread across the 36 character
- Character 'V' has the highest number of samples at 7, while 'E' and 'O' have the next highest number of samples at 6 each while '8', 'F', 'N', 'P' have the lowest number of samples at 1 each. Given the shape and structure of 'E" and 'O' training directly on just the given samples wouuld not be sufficient as other characters like '0', 'B', 'F', and 'P' could easily be misclassified as 'O' and 'E' respectively.

| Character | Count | | Character |Count  | |Character | Count | |Character | Count |
|-----------|-------|-|-----------|-------|-|----------|-------|-|----------|-------|
| 0         | 2     | | A         | 4     | | K        | 3     | | U        | 2     |
| 1         | 5     | | B         | 3     | | L        | 3     | | V        | 7     |
| 2         | 5     | | C         | 5     | | M        | 5     | | W        | 4     |
| 3         | 2     | | D         | 5     | | N        | 1     | | X        | 2     |
| 4         | 2     | | E         | 6     | | O        | 6     | | Y        | 2     |
| 5         | 3     | | F         | 1     | | P        | 1     | | Z        | 5     |
| 6         | 3     | | G         | 5     | | Q        | 5     |
| 7         | 4     | | H         | 4     | | R        | 3     |
| 8         | 1     | | I         | 2     | | S        | 4     |
| 9         | 5     | | J         | 3     | | T        | 2     |

#### Augmentation 🔄
Due to the limited number and unbalance between the number of samples for each character, data augmentation was performed to increase the number of training samples. The following augmentation techniques were used leveraging the Albumentations library

- RandomBrightnessContrast
    - brightness_limit=(-0.1,0.1)
    - contrast_limit=(-0.1,0.1)
    - p=0.05
- GaussNoise
    - var_limit=(5.0,10.0)
    - p=0.2
- PixelDropout
    - dropout_prob=0.01
    - per_channel=False
    - p=0.25

Including the original samples a total of 100 augmented samples were generated for each character.
All training images can be found under ./trainingdata/singleChar_Augment

## Tesseract 👁️
As an immediate available algorithm Tesseract was used to extract the text from the captcha images. Read more about Tesseract [here](https://github.com/tesseract-ocr/tesseract).

### Configuration ⚙️
- psm 10: Treat each cropped region as a single character
- oem 3: Use the defa ult OCR engine
- tessedit_char_whitelist: Limit the characters to be recognized to uppercase alphabets and numerals

```bash
--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJLKMNOPQRSTUVWXYZ0123456789
```

### Installation 📥
Download and install Tesseract

- **Windows**: Download the [Tesseract Installer](https://github.com/UB-Mannheim/tesseract/wiki)
- **Mac**: `brew install tesseract`
- **Linux**: `sudo apt install tesseract-ocr`om

## Future Areas of Improvements 🗺️
- Leverage OpenCV to detect the contours of the characters within the Captcha and extract it out as an individual region
- Enable ensemble output to allow users to have more flexibility to choose between the results from CNN and Tesseract
- Flexibility in handling images with different number of characters returned
- Generation of synthetic test & validation data for accuracy
- Connecting with platforms like weights and biases (wand) for MLOPs which would provide better model version handling and loading
- Alternative OCR services like AWS Textract would also provide additional options
