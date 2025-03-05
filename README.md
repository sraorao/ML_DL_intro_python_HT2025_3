# Introduction to Machine Learning and Deep Learning in Python (2-Day Course)

## By [Irina Chelysheva](https://github.com/Chelysheva) and [Srinivasa Rao Rao](https://github.com/sraorao) ([Oxford Profile](https://www.nds.ox.ac.uk/team/srinivasa-rao))

## Course Aim

This course provides a practical introduction to applying machine learning and deep learning algorithms to biological data using the Python programming language. Through a blend of lectures and hands-on training, participants will learn the steps of data processing, preparation, and application of machine learning techniques to solve various problems, guided by expert tutors. By the end of the course, participants should be able to apply machine learning and deep learning techniques to both tabular and image data.

### Session Details

You are expected to attend two half-day sessions:

- **Monday, 03 March**: 10:00 - 13:00
- **Tuesday, 05 March**: 10:00 - 13:00

## Requirements
- Python 3.8 or above
- A Python IDE (Spyder, Jupyter, VS Code, or similar)
- Python packages: `scikit-learn`, `pytorch`, `matplotlib`, `pandas`, `seaborn`, `opencv`, `pillow`
- If you have conda/mamba installed, the easiest thing is to create a new environment with the provided yaml file:
```
conda env create -f deeplearning.yaml
conda activate deeplearning
```
## Course Objectives

By the end of this course, you will learn:

- How to pre-process your data for machine learning and deep learning analysis.
- How to apply machine learning techniques to address biological questions using example datasets.
- How to apply deep learning on an example image dataset.
- How to evaluate and visualize analysis results.

## Course Format

The course is structured into two half-day sessions as follows:

### Day 1

**Theory:**
- Introduction to Machine Learning
- Types of Problems That Can Be Resolved with Machine Learning
- Overview of Machine Learning Algorithms
- Feature Selection Algorithms

**Hands-on 1:**
- Application of machine learning in Python using the `sklearn` library on a publicly available clinical dataset (solving a classification problem).

**Hands-on 2:**
- Application of feature selection and machine learning in Python using the `sklearn` library on a publicly available socio-demographical medical dataset (solving a regression problem).

### Day 2

**Theory:**
- Introduction to Image Analysis with Deep Learning
- Use Cases
- Image Pre-processing

**Hands-on:**
- Pre-processing images in Python using `NumPy` and `OpenCV` libraries.

**Theory:**
- Convolutional Neural Networks (CNNs) using `PyTorch`

**Hands-on:**
- Application of CNNs on an image classification problem (Acute Lymphoblastic Leukemia classification).
- Model Evaluation
- Using Pre-trained Models
