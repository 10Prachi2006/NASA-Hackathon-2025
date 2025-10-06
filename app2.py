import matplotlib
matplotlib.use('Agg')  # for headless server

import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from fpdf import FPDF

app = Flask(__name__)

@app.route('/run-model', methods=['POST'])
def run_model():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    try:
        df = pd.read_csv(file)
        columns = [col for col in ['koi_period', 'koi_prad', 'koi_disposition'] if col in df.columns]
        if len(columns) < 3:
            return jsonify({'error': 'Required columns missing'}), 400
        df = df[columns].dropna()
        X = df[['koi_period', 'koi_prad']]
        y = df['koi_disposition']
        y = y.map({'CONFIRMED': 1, 'FALSE POSITIVE': 0, 'CANDIDATE': 2})
        mask = y.notnull()
        X = X[mask]
        y = y[mask]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        timestamp = int(time.time())

        # Confusion Matrix Plot
        plt.figure(figsize=(6,4))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        img_name = f'confusion_matrix_{timestamp}.png'
        img_path = f'/home/Prachi10/static/{img_name}'
        plt.savefig(img_path)
        plt.close()

        # Feature Importances
        importances = model.feature_importances_
        features = X_train.columns.tolist()
        plt.figure(figsize=(6,3))
        sns.barplot(x=importances, y=features)
        plt.title("Feature Importances")
        plt.tight_layout()
        imp_imgname = f"feature_importances_{timestamp}.png"
        imp_imgpath = f"/home/Prachi10/static/{imp_imgname}"
        plt.savefig(imp_imgpath)
        plt.close()

        # ROC Curve (Binary only)
        roc_imgname = None
        if len(set(y_test)) == 2:
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend(loc="lower right")
            roc_imgname = f"roc_curve_{timestamp}.png"
            roc_imgpath = f"/home/Prachi10/static/{roc_imgname}"
            plt.savefig(roc_imgpath)
            plt.close()

        # Classification Report as string (for PDF)
        class_report = classification_report(y_test, y_pred, output_dict=False)

        # Generate PDF with FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Exoplanet ML Report", ln=1, align='C')
        pdf.cell(200, 10, txt=f"Accuracy: {acc}", ln=2)
        pdf.ln(5)
        pdf.set_font("Arial", size=10)
        pdf.cell(200, 10, txt="Classification Report:", ln=3)
        pdf.ln(5)
        for line in class_report.split('\n'):
            pdf.cell(0, 5, txt=line, ln=1)
        pdf.ln(5)
        pdf.cell(200, 10, txt="See attached plots for details.", ln=4)

        pdf_path = f"/home/Prachi10/static/report_{timestamp}.pdf"
        pdf.output(pdf_path)

        output = {
            'accuracy': acc,
            'image_url': f'https://prachi10.pythonanywhere.com/static/{img_name}',
            'feature_importances_url': f'https://prachi10.pythonanywhere.com/static/{imp_imgname}',
            'roc_curve_url': f'https://prachi10.pythonanywhere.com/static/{roc_imgname}' if roc_imgname else None,
            'pdf_url': f'https://prachi10.pythonanywhere.com/static/report_{timestamp}.pdf'
        }
        return jsonify(output)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return "Exoplanet Model API is live! Use POST to /run-model with your CSV file."

@app.route('/static/<filename>')
def serve_static(filename):
    return send_from_directory('/home/Prachi10/static', filename)

# # Remove for deployment in PythonAnywhere; only use locally:
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)
