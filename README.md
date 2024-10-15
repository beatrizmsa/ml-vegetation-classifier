# Vegetation Classification using Machine Learning

## Project Overview

The aim of this project is to apply various machine learning methods to a dataset containing environmental and geographical features to classify different types of vegetation. The dataset includes features such as altitude, slope, proximity to water, canopy density, rainfall, and more, each corresponding to a specific region. The target variable is the vegetation type, which consists of three assigned classes representing different ecosystems.
## Dataset

The dataset contains various features related to geographic and environmental properties of different land regions. Key features include:

- **Altitude** (in meters)
- **Slope** (in degrees)
- **Horizontal/Vertical Distance to Water** (in meters)
- **Distance to Roadways** (in meters)
- **Shadow Index** (measured at 9h, 12h, 15h)
- **Canopy Density** (percentage of land covered by trees)
- **Rainfall in Summer/Winter** (in mm)
- **Wind Exposure Level** (in Km/h)
- **Soil Type** (40 different soil types)
- **Wilderness Area** (4 different wilderness areas)
- **Vegetation Type** (Target variable with 7 classes)

### Class Distribution

The dataset is imbalanced, with the following distribution:

| Vegetation Type | Type 1 | Type 2 | Type 3 | Type 4 | Type 5 | Type 6 | Type 7 |
|-----------------|--------|--------|--------|--------|--------|--------|--------|
| Samples         | 2160   | 1404   | 1620   | 1080   | 1944   | 2160   | 864    |

We will be working with the classes **type 1**,**type 3**,**type 6**.

## Objectives

The project consists of the following main tasks:

1. **Exploratory Data Analysis (EDA)**
   - Perform descriptive statistics and analyze feature distributions.
   - Conduct bivariate analysis to explore correlations between features and vegetation types.

2. **Model Training**
   - Apply the following machine learning methods to the data:
     - Logistic Regression
     - Linear Discriminant Analysis (LDA)
     - Quadratic Discriminant Analysis (QDA)
   - Use various resampling techniques, including:
     - Holdout
     - 5-fold and 10-fold Cross Validation
     - Leave-One-Out Cross Validation (LOOCV)
     - Bootstrap
   - Evaluate models using metrics like accuracy, precision, recall, and F1-score.

3. **Feature Selection**
   - Investigate if reducing the number of features (via regularization) improves model performance.

## Technologies Used
- Python 3.11
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

## Project Structure

- **data/**: Contains the data for each assigned class.
- **notebook.ipynb**: Jupyter notebook for this project.
- **requirements.txt**: List of dependencies required for running the project.
- **README.md**: Overview of the project.

## Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/beatrizmsa/ml-vegetation-classifier.git
   ```

2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebook.

## Results

## Conclusion

## Authors

- [Alexandre Marques - 1240435](https://github.com/AlexandreMarques27)
- [Beatriz Sá - 1240440](https://github.com/beatrizmsa)
- [Diogo Gaspar - 1200966](https://github.com/diogogaspar123)

