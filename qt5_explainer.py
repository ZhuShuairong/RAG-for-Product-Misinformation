from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLineEdit, QLabel, QComboBox, QCheckBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pandas as pd
import re
from scripts.build_retriever import Retriever
import sys


class ExplainerReviewApp(QWidget):
    def __init__(self, model_dir, product_info_file, max_length=512):
        super().__init__()

        # Load model and tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(self.device)

        # Load product data (this should come from your actual product data)
        self.product_data = self.load_product_data(product_info_file)
        self.product_names = self.product_data["product_name"].tolist()

        self.max_length = max_length

        # Setup the GUI
        self.initUI()

    def load_product_data(self, file_path):
        """Load product data from a CSV file"""
        return pd.read_csv(file_path, dtype=str, low_memory=False).sort_values(by="product_name", ascending=True)

    def initUI(self):
        self.setWindowTitle('Fake Review Explainer')

        # Set the layout
        layout = QVBoxLayout()

        # Create a text input box for review title
        self.review_title_input = QLineEdit(self)
        self.review_title_input.setPlaceholderText('Enter your review title here...')
        self.review_title_input.setFixedHeight(40)
        self.review_title_input.setFixedWidth(700)
        layout.addWidget(self.review_title_input)

        # Create a text input box for review context
        self.comment_input = QLineEdit(self)
        self.comment_input.setPlaceholderText('Enter your review here...')
        self.comment_input.setFixedHeight(100)
        self.comment_input.setFixedWidth(700)
        layout.addWidget(self.comment_input)

        # Create a combo box to select product
        self.product_select = QComboBox(self)
        self.product_select.addItems(self.product_names)  # Add product names from product data
        self.product_select.setFixedWidth(700)
        layout.addWidget(self.product_select)

        # Create a checkbox to select "Recommended"
        self.recommended_checkbox = QCheckBox("Recommended", self)
        layout.addWidget(self.recommended_checkbox)

        # Create a button to run the detection
        self.detect_button = QPushButton('Check Review', self)
        self.detect_button.setFixedHeight(80)
        self.detect_button.setFixedWidth(700)
        self.detect_button.clicked.connect(self.on_check_click)
        layout.addWidget(self.detect_button)

        # Label to display the result
        self.result_label = QLabel('Result: None', self)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFont(QFont('Arial', 24))  # Set a larger font for result
        self.result_label.setFixedHeight(100)
        self.result_label.setFixedWidth(700)
        layout.addWidget(self.result_label)

        # Reason label to display the explanation
        self.reason_label = QLabel('Reason: None', self)
        self.reason_label.setAlignment(Qt.AlignLeft)  # Left align reason
        self.reason_label.setWordWrap(True)  # Enable word wrap
        self.reason_label.setFont(QFont('Arial', 16))  # Adjust font size for reason text
        self.reason_label.setFixedHeight(150)
        self.reason_label.setFixedWidth(700)
        layout.addWidget(self.reason_label)

        # Set the layout
        self.setLayout(layout)

        # Make the layout resize with the window size
        layout.setContentsMargins(20, 20, 20, 20)

    def on_check_click(self):
        # Get the input review title and context
        review_title = self.review_title_input.text()
        review_text = self.comment_input.text()

        # Get the selected product information
        selected_product_name = self.product_select.currentText()
        selected_product_info = self.product_data[self.product_data["product_name"] == selected_product_name]

        # Determine the recommended status based on checkbox
        recommended_status = 1.0 if self.recommended_checkbox.isChecked() else 0.0

        # Concatenate review title, review text, and product information to form the context
        context = (
            f"Review title: {review_title} | "
            f"Review text: {review_text} | "
            f"Rating: {selected_product_info['rating'].values[0]} | "
            f"Recommended: {recommended_status} | "
            f"Product summary: {selected_product_info['product_name'].values[0]} by {selected_product_info['brand_name'].values[0]}. "
            f"Highlights: {selected_product_info['highlights'].values[0]}."
        )

        # Process the review with the model
        label, reason = self.predict_fake_review_with_explainer(context)

        if label == "Fake":
            # label_text = "Fake Review"
            label_text = "Useless Review"
        else:
            # label_text = "Real Review"
            label_text = "Useful Review"

        # Display the result
        # self.result_label.setText(f"Label: {label}")
        self.result_label.setText(f"Label: {label_text}")

        # Display the explanation
        self.reason_label.setText(f"Reason: {reason}")

        # Set color for result text based on fake or real
        if label == "Real Review":
            self.result_label.setStyleSheet("color: green;")  # Green for real
        else:
            self.result_label.setStyleSheet("color: red;")  # Red for fake

    def predict_fake_review_with_explainer(self, review_text):
        # Tokenize the review input
        inputs = self.tokenizer(review_text, return_tensors='pt', truncation=True, padding=True, max_length=self.max_length)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Model prediction
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=64,
                num_beams=4,
                length_penalty=1.0,
                early_stopping=True
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the label and reason from the decoded output
        label, reason = self.parse_label_and_reason(decoded)

        return label, reason

    def parse_label_and_reason(self, output_text):
        """
        Extract the label (fake/real) and reason from the generated text.
        The format is expected to be like:
        "Label: real\nReason: [explanation text]"
        """
        label = None
        reason = None

        # Attempt to extract the label and reason from the model's generated output
        match_label = re.search(r"label:\s*(fake|real)", output_text, re.IGNORECASE)
        match_reason = re.search(r"reason:\s*(.*)", output_text, re.IGNORECASE)

        if match_label:
            label = match_label.group(1).capitalize()  # Extract the label and capitalize it
        if match_reason:
            reason = match_reason.group(1)  # Extract the reason text

        # Fallback if either is missing
        if not label:
            label = "fake"
        if not reason:
            reason = "The reason for this label is unclear."

        return label, reason


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ExplainerReviewApp(
        model_dir='models/bart_fake_explainer',  # Path to your trained explainer model
        product_info_file='data/product_info.csv',  # Path to your product info CSV
        max_length=512,  # Max length for tokenizer
        # max_length=384,
    )

    # Set the overall window size larger (5x) and make it resizable
    ex.setMinimumSize(700, 500)  # Minimum size for resizing
    ex.setMaximumSize(1000, 800)  # Maximum size for resizing
    ex.show()

    sys.exit(app.exec_())
