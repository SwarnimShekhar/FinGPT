import unittest
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

class TestFinGPTAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_name = "finGPT-audit-model"  # Replace with model version if necessary
        cls.model = T5ForConditionalGeneration.from_pretrained(model_name)
        cls.tokenizer = T5Tokenizer.from_pretrained(model_name)
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.model.to(cls.device)

    def test_prediction_output(self):
        text = "The company reported record profits in Q1 2025."
        input_text = "classify: " + text
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        output_ids = self.model.generate(**inputs)
        output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        self.assertIn(output, ["positive", "negative", "neutral"])

    def test_empty_input(self):
        text = ""
        input_text = "classify: " + text
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        output_ids = self.model.generate(**inputs)
        output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        self.assertTrue(isinstance(output, str))

if __name__ == "__main__":
    unittest.main()