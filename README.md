# Pattern Mining Toolkit

The Pattern Mining Toolkit is a Python library designed for various pattern mining algorithms. It provides implementations for several well-known algorithms, making it easier to analyze and extract patterns from text data.

## Features

- **Apriori Algorithm**: Efficiently finds frequent itemsets in transactional data.
- **FP-Growth Algorithm**: An efficient method for mining frequent patterns without candidate generation.
- **Maximal Pattern Mining**: Identifies maximal patterns from frequent itemsets.
- **Sequential Pattern Mining**: Extracts sequential patterns from text sequences.
- **Graph-based Pattern Mining**: Analyzes text data using graph theory.
- **Closed Pattern Mining**: Finds closed patterns that cannot be extended without losing support.
- **ECLAT Algorithm**: A vertical data format approach for frequent itemset mining.
- **CHARM Algorithm**: Mines closed itemsets efficiently.
- **SPADE Algorithm**: Sequential pattern discovery using equivalence classes.
- **PrefixSpan Algorithm**: A pattern mining algorithm for sequential data.

## Installation

To install the required dependencies, run:

```
pip install -r requirements.txt
```

## Usage

To use the Pattern Mining Toolkit, you can import the `PatternMiningToolkit` class from the `pattern_mining_toolkit` module:

```python
from src.pattern_mining_toolkit import PatternMiningToolkit

# Create an instance of the toolkit
toolkit = PatternMiningToolkit()

# Load your text data
texts = ["sample text data", "another sample text"]
toolkit.load_text_data(texts)

# Run the Apriori algorithm
frequent_itemsets = toolkit.apriori_algorithm(min_support=0.3)
print(frequent_itemsets)
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.