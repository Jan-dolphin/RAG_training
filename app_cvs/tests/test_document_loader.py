import unittest
import shutil
import tempfile
import os
from pathlib import Path
from src.document_loader import CVDocumentLoader
from src.models import Candidate

class TestCVDocumentLoader(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        self.loader = CVDocumentLoader(self.test_dir)

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def create_dummy_file(self, filename, content="This is some dummy content for testing purposes."):
        path = Path(self.test_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return path

    def test_init_invalid_directory(self):
        with self.assertRaises(ValueError):
            CVDocumentLoader("non_existent_directory_12345")

    def test_get_candidate_name(self):
        # Using helper method directly
        path = Path(self.test_dir) / "John_Doe_CV_EN.docx"
        name = self.loader._get_candidate_name_from_path(path)
        self.assertEqual(name, "John Doe")

        path2 = Path(self.test_dir) / "Jane_Smith_CV.pdf"
        name2 = self.loader._get_candidate_name_from_path(path2)
        self.assertEqual(name2, "Jane Smith")

        path3 = Path(self.test_dir) / "simple_name.txt"
        name3 = self.loader._get_candidate_name_from_path(path3)
        self.assertEqual(name3, "simple name")

    def test_load_txt_file(self):
        self.create_dummy_file("test_candidate.txt", "Some relevant experience with Python and AI. This text needs to be longer than 50 characters to be loaded.")
        candidate = self.loader.load_single_cv(str(Path(self.test_dir) / "test_candidate.txt"))
        
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.name, "test candidate")
        self.assertIn("Python", candidate.full_cv_text)
        self.assertEqual(candidate.metadata['extension'], '.txt')

    def test_load_csv_file(self):
        csv_content = "name,experience\nAlice,5 years java\nBob,3 years python"
        self.create_dummy_file("candidates.csv", csv_content)
        
        # CSVLoader typically loads each row as a document. 
        # Our loader combines them.
        candidate = self.loader.load_single_cv(str(Path(self.test_dir) / "candidates.csv"))
        
        self.assertIsNotNone(candidate)
        self.assertIn("Alice", candidate.full_cv_text)
        self.assertIn("Bob", candidate.full_cv_text)
        self.assertEqual(candidate.metadata['extension'], '.csv')

    def test_recursive_loading(self):
        self.create_dummy_file("root.txt", "Root file content." * 5)
        self.create_dummy_file("level1/nested.txt", "Nested file content." * 5)
        self.create_dummy_file("level1/level2/deep.txt", "Deep file content." * 5)
        
        candidates = self.loader.load_all_cvs()
        
        self.assertEqual(len(candidates), 3)
        names = sorted([c.name for c in candidates])
        self.assertEqual(names, ["deep", "nested", "root"])

    def test_ignore_small_files(self):
        self.create_dummy_file("small.txt", "Too short")
        candidates = self.loader.load_all_cvs()
        self.assertEqual(len(candidates), 0)

    def test_unsupported_format(self):
        self.create_dummy_file("image.png", "not text")
        # Should just be ignored by load_all_cvs loop because of extension check
        candidates = self.loader.load_all_cvs()
        self.assertEqual(len(candidates), 0)
        
        # Explicit load should return None or raise
        # The code catches Exception and logs error, returns None
        candidate = self.loader.load_single_cv(str(Path(self.test_dir) / "image.png"))
        self.assertIsNone(candidate)

if __name__ == '__main__':
    unittest.main()
