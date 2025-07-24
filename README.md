# BiasDF
Official implementation of the paper 'Bias Analysis for Synthetic Face Detection: A Case Study of the Impact of Facial Attributes'
===========================================
Facial Attribute Combination Generator
===========================================

This Python script generates a CSV file with:
- All possible combinations of facial attributes
- One-hot encoded binary features
- A natural-language description of each combination

-------------------------
📦 Requirements
-------------------------

- Python 3.6 or higher
- pandas

To install required package:

    pip install pandas

-------------------------
🚀 How to Use the Script
-------------------------

Run the script with the --output argument to specify where to save the generated CSV file.

Example:

    python generate_input_description.py --output ./df_input_description.csv

This will generate a file named "df_input_description.csv" in the current directory.

-------------------------
📂 Output Format
-------------------------

The output CSV file will contain:
- One-hot encoded binary columns for each attribute value (e.g., man, woman, black_hair, gray_hair, etc.)
- A "description" column that combines all the attributes in a readable sentence.

Example row in the output file:

    no attractive woman and old and gray hair and wavy hair and white skin and green eyes and Mustach and heavy makeup and pointy nose and square face

-------------------------
🛠 Customization
-------------------------

To add or remove attributes, open the Python script `generate_input_description.py` and edit the dictionary called `attributes` at the top of the file.

For example, to add a new category called "Emotion", you could add:

    "Emotion": ["happy", "sad", "angry"]

Don't forget to adjust the logic if needed.

-------------------------
📁 File Structure
-------------------------

- generate_input_description.py : the main script
- README.txt                   : this file
- df_input_description.csv     : generated output after running the script

-------------------------
📧 Contact
-------------------------

For feedback or questions, please reach out to the author.
