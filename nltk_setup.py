# nltk_setup.py
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download all required NLTK data
resources = [
    'punkt',
    'stopwords',
    'wordnet',
    'averaged_perceptron_tagger',
    'maxent_ne_chunker',
    'words'
]

for resource in resources:
    try:
        nltk.download(resource)
        print(f"Successfully downloaded {resource}")
    except Exception as e:
        print(f"Error downloading {resource}: {e}")
