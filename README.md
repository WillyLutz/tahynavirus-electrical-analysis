# Electrical signal analysis : Tahynavirus infected brain slices

## Description
Its aim is to provide signal processing and machine learning solutions for electrical signal analysis. 
In this specific case it has been used on human brain slices. It allows the user to use different analysis and 
data processing procedures. In those you can find smoothing, Fast-Fourier-Transform, data augmentation algorithms and others.
Those procedures have been optimized for this very project in this repository, so you may want to adapt it in many ways for your own usage.

For more information on the possible usages, please refer to the [corresponding section](#usage).

You can also check out [other repositories with a similar use](#other-repositories-with-similar-use)


## Development context
This project is developed in the context of public research in biology. It has been developed as support for a publication not disclosed here.

## Visuals and resulting figures
On this part you will have a quick overview on the different resulting figures possible with this project. They will be given without context and are for 
illustration purpose only, and may not be relevant with the actual article this github is from.

### Frequency/power plots
Plot your signal in the frequency domain.

<img src="https://github.com/WillyLutz/electrical-analysis-sars-cov-2/blob/main/Figures_README/smoothened_frequencies.png" width=250 height=250>

### Amplitude barplot
Compute the average power and variation of different labels. Allows a restriction to a specific frequency range.

<img src="https://github.com/WillyLutz/electrical-analysis-sars-cov-2/blob/main/Figures_README/barplot_amplitude.png" width=250 height=250>

### 2D PCA plot
Fit a Principal Component Analysis on you data and plot it in a two-dimensionnal space. You can also decide to fit the
model only on a few label, then apply the transformation to another !

<img src="https://github.com/WillyLutz/electrical-analysis-sars-cov-2/blob/main/Figures_README/pca_2D.png" width=250 height=250>

### 3D PCA plot
Fit a Principal Component Analysis on you data and plot it in a three-dimensionnal space. You can also decide to fit the
model only on a few label, then apply the transformation to another !

<img src="https://github.com/WillyLutz/electrical-analysis-sars-cov-2/blob/main/Figures_README/PCA_3D.png" width=250 height=250>

### Confusion matrix
Used to check the performance of a machine learning model, here Random Forest Classifier. You can train on specific label
and test you model on different ones, to see where the model classify them among the training labels.

<img src="https://github.com/WillyLutz/electrical-analysis-sars-cov-2/blob/main/Figures_README/confusion_matrix.png" width=600 height=200>

### Feature importance
Plot the relative importance of the features for a trained RFC model.

<img src="https://github.com/WillyLutz/electrical-analysis-sars-cov-2/blob/main/Figures_README/feature_importance.png" width=250 height=250> 

## Data acquisition
The signal has been recorded at 10 kHz, with a MEA 60 channels electrode. 
For more information about the array, refer to [their page](https://www.multichannelsystems.com/products/60-3dmea2001280ir-ti-gr). Each recording has been 
done 3 times,on a minimum of 3 brain slices per test batch.


## Data formatting 
For most (if not all) of the analysis, a certain data format will be needed. Any modification on the data format may 
induce errors and bugs for the good use of the project.


### Project organization
```bash
├── TAHYNAVIRUS
│   ├── scripts
│   │   ├── complete_procedures.py
│   │   ├── data_processing.py
│   │   ├── machine_learning.py
│   │   ├── main.py
│   │   ├── PATHS.py
│   │   ├── signal_processing.py
│   ├── venv

```

### Organizing the data
To use efficiently the project, a certain architecture will be needed when organizing the data.
```bash
├── base
│   ├── DATASET
│   ├── MODELS
│   ├── RESULTS
│   │   ├── Figures README Paper
│   │   │   ├──myfigures.png
│   │   ├──myfigures.png
│   ├── DATA
│   │   ├── drug condition*
│   │   │   ├── recording time**
│   │   │   │   ├── cell condition***
│   │   │   │   │   ├── samples****
│   │   │   │   │   │   ├── myfiles.csv*****
```
* E.g. '-RG10b', '+RG10b'

** Must follow the format T=[time][H/MIN]. E.g. 'T=48H', 'T=0MIN', 'T=30MIN'.

*** What you want to classify. E.g. 'TAHV', 'NI'. 

**** The sample number. E.g. '1', '2'... 

***** The files that contain the data. They must follow a [certain format](#data-format).

In the data folder, you can multiply every directory as much as you have conditions.

### Data format
Across all the analysis, multiple data type will be generated. For all the files generated, it is recommended to keep tracks of the
different conditions of this very data in the file name.

#### Raw file
Usually of format similar as following:

```2022-09-16T14-02-25t=48h NI1 NODRUG_D-00145_Recording-0_(Data Acquisition (1);MEA2100-Mini; Electrode Raw Data1)_Analog.csv```

This type of file contains the raw output in csv format of the electrode array. It may look as following:
<img src="https://github.com/WillyLutz/electrical-analysis-sars-cov-2/blob/main/Figures_README/sshot_raw_file_sarscov.png" width=950 heigth=320>

#### Processed format
Usually of format similar as following: 
```pr_2022-09-16T14-03-55.csv```

In fact it is equivalent to the raw file with only th data (beheaded of the information headlines).
It may look like:

<img src="https://github.com/WillyLutz/electrical-analysis-sars-cov-2/blob/main/Figures_README/sshot_pr_file_sarscov.png" width=950 heigth=230>

The column headers are normalized to function with the project. Other headers will not function without modifying directly the code.

#### Frequencies format
Usually a format similar as following: 
```freq_50hz_sample29_2022-09-16T14-05-24.csv```

It is the result of the Fast Fourier Transform applied on the average signal across the channels (after channel selection) from the processed files. It mayt look like this:

<img src="https://github.com/WillyLutz/electrical-analysis-sars-cov-2/blob/main/Figures_README/sshot_freq_file_sarscov.png" width=200 heigth=300>

The column headers 'mean' and 'Frequency [Hz]' are normalized to function with the project. Other headers will not function without modifying directly the code.

## Development specification
Language: Python 3.10

OS: Ubuntu 22.04.1 LTS

## Usage
This part will help you getting started with the use of this project. Please note that this project has heavy dependance on the python package `fiiireflyyy`, developed by the same author. 

#### The PATHS.py file
After succesfully cloning the repository as an IDE project, the first thing you want to do is to modify the constants used for this project, such as the absolute paths. 

Go to the file `PATHS.py`.
```python
DISK = "/media/wlutz/TOSHIBA EXT/Electrical activity analysis/TAHYNAVIRUS/DATA"
```

This first path is what is referred as `base` in the [data organisation](#organizing-the-data). Replace it by your own absolute path. From the `DISK` path, each and every data used or generated by the project will be under it.


## Support
For any support request, you can either use this project issue tracker, or state your request at <willy.lutz@irim.cnrs.fr> by precising in the object the name of this repository followed by the summarized issue.

## Contributing
This project is open to any suggestion on the devlelopment and methods. 

## Authors and acknowledgment
Author: Willy LUTZ

Principal investigator: Raphaël Gaudin

Context: MDV Team, IRIM, CNRS, Montpellier 34000, France.

## License
This open source project is under the MIT Licence.

MIT License

Copyright (c) [2023] [Electrical signal analysis : SARS-CoV-2 infected organoids]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Project status
on going

## Other repositories with similar use
From the same author:
- no information disclosed for the moment 
