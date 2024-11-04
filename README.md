# MTHAEL
MTHAEL: Cross-Architecture IoT Malware Detection Based on Neural Network Advanced Ensemble Learning

Project Overview

This project focuses on detecting malware threats in IoT devices by using an ensemble of deep learning architectures. By leveraging multiple models—including Convolutional Neural Networks (CNNs), Fully Convolutional Networks (FCN), and Recurrent Neural Networks (RNN)—this approach combines the strengths of various models to improve the accuracy and robustness of IoT malware detection.

Key Features

	•	Ensemble Learning: Combines multiple deep learning models (e.g., 1D CNN, FCN, RNN) to achieve a higher detection accuracy for IoT-based malware threats.
	•	Visualization and Analysis: Includes detailed ROC and confusion matrices to evaluate model performance and visualize the effectiveness of the ensemble model.
	•	IoT-Specific Data: Trained on IoT malware data to ensure the model is tailored to common IoT device threats, making it suitable for real-world cybersecurity applications.
	•	Efficient Threat Detection: Designed to detect malware effectively across a range of IoT malware categories, ensuring minimal false positives and high detection rates.

Project Structure

	•	1DCNN.ipynb: Notebook implementing a 1D CNN architecture for IoT malware detection.
	•	Lstm_Conv1D.ipynb: Notebook for a model combining LSTM and Conv1D layers.
	•	Ensemble Notebooks: Main ensemble configurations combining CNN, FCN, and RNN architectures to enhance performance.
	•	ROC and Confusion Matrices: Precomputed ROC curves and confusion matrices (e.g., ROC for MTHEL.png, CM-Ensemble.csv) showing the ensemble model’s performance.
	•	Model Files: Serialized model files (e.g., FCN.pkl, RNN.pkl) for the FCN, RNN, and other components of the ensemble.
	•	Dataset and History: Contains data history files, including MTHEL-History.csv, tracking model performance over training epochs.

Technical Requirements

	•	TensorFlow and Keras: Core libraries for building and training the deep learning models.
	•	Scikit-learn: Used for data handling and metrics computation.
	•	Matplotlib and Seaborn: For visualizing ROC curves and confusion matrices.
	•	Pandas and Numpy: For data preprocessing and management.

How It Works

	1.	Data Preparation: IoT malware data is preprocessed and fed into various deep learning models.
	2.	Model Training: Individual models, such as 1D CNN, FCN, and RNN, are trained on the IoT malware dataset.
	3.	Ensemble Method: Predictions from the models are combined in an ensemble setup to produce a more robust classification output.
	4.	Evaluation: The ensemble model’s performance is assessed using ROC curves, confusion matrices, and accuracy scores to determine its effectiveness in detecting IoT malware threats.

Results and Findings

	•	Enhanced Detection Accuracy: The ensemble approach achieves better performance than individual models, as demonstrated by high ROC-AUC scores and detailed confusion matrices.
	•	Robustness Against Variations: The ensemble method provides a resilient detection mechanism, capable of identifying various IoT malware families effectively.
	•	Analysis Notebooks: The repository includes multiple notebooks with visualizations (e.g., Graphs(ALL).ipynb), showcasing detailed metrics and the effectiveness of the ensemble method.

Conclusion

This project provides an effective solution for IoT malware threat hunting by combining multiple deep learning architectures in an ensemble. The approach improves detection accuracy, reduces false positives, and enhances the robustness of IoT security solutions.

Cite:

@ARTICLE{9165209,
  author={Vasan, Danish and Alazab, Mamoun and Venkatraman, Sitalakshmi and Akram, Junaid and Qin, Zheng},
  journal={IEEE Transactions on Computers}, 
  title={MTHAEL: Cross-Architecture IoT Malware Detection Based on Neural Network Advanced Ensemble Learning}, 
  year={2020},
  volume={69},
  number={11},
  pages={1654-1667},
  keywords={Malware;Computer architecture;Feature extraction;Machine learning;Neural networks;Robustness;Training;Internet-of-Things;malware threat hunting;robust malware detection;advanced ensemble learning;cross-architectures},
  doi={10.1109/TC.2020.3015584}}
