import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLineEdit, QLabel, QComboBox, QCheckBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from transformers import AutoTokenizer, RobertaForSequenceClassification
import torch
import json
import pandas as pd
from scripts.build_retriever import Retriever

class FakeReviewApp(QWidget):
    def __init__(self):
        super().__init__()

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("models/roberta_fake")
        self.model = RobertaForSequenceClassification.from_pretrained("models/roberta_fake")
        self.retriever = Retriever(model_dir="models/all-MiniLM-L6-v2")

        # Load product data (this should come from your actual product data)
        self.product_data = self.load_product_data("data/product_info.csv")
        self.product_names = self.product_data["product_name"].tolist()

        # Setup the GUI
        self.initUI()

    def load_product_data(self, file_path):
        """Load product data from a CSV file"""
        return pd.read_csv(file_path, dtype=str, low_memory=False).sort_values(by="product_name", ascending=True)

    def initUI(self):
        self.setWindowTitle('Fake Review Detection')

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
            f"Highlights: {selected_product_info['highlights'].values[0]}. "
            f"Ingredients: {selected_product_info['ingredients'].values[0]}"
        )

        # Retrieve relevant product information using RAG
        retrieved_docs = self.retriever.retrieve(context, top_k=3)

        # Process the review with the model
        result = self.predict_fake_review(context, retrieved_docs)

        # Display the result
        self.result_label.setText(f"Result: {result}")

        # Set color for result text based on fake or real
        if result == "Real Review":
            self.result_label.setStyleSheet("color: green;")  # Green for real
        else:
            self.result_label.setStyleSheet("color: red;")  # Red for fake

    def predict_fake_review(self, review_text, retrieved_docs):
        # Tokenize the review input
        inputs = self.tokenizer(review_text, return_tensors='pt', truncation=True, padding=True, max_length=256)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Model prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()

        # Retrieve product metadata
        fact_label = self.detect_factual_mismatch(review_text, retrieved_docs)

        # Combine the prediction with factual consistency check
        if prediction == 1 and fact_label == "real":
            return "Real Review"
        else:
            return "Fake Review"

    def detect_factual_mismatch(self, review_text, retrieved_docs):
        """Use RAG to detect factual mismatches between the review and retrieved product info."""
        text = review_text.lower()
        for doc in retrieved_docs:
            doc_text = doc["document"].lower()
            # Here you can add more checks as required
            if doc_text not in text:
                return "fake"
        return "real"


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FakeReviewApp()

    # Set the overall window size larger (5x) and make it resizable
    ex.setMinimumSize(700, 500)  # Minimum size for resizing
    ex.setMaximumSize(1000, 800)  # Maximum size for resizing
    ex.show()

    sys.exit(app.exec_())
