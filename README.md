# SEC Filing Sentiment Analyzer

## Overview
A financial sentiment analysis tool leveraging FinBERT to automatically analyze SEC filings and extract key insights. The project aims to provide nuanced sentiment analysis of financial documents, helping identify significant trends and patterns that might be missed in manual review.

- Added a zipfile that contains all the nescesasry files. Uncompress the stockbert.rar file and you can run the program from main.py by typing python main.py in the terminal.
- ![image](https://github.com/user-attachments/assets/d09e744e-6cf0-4eeb-974a-ec783c335195)
- Running the program will also create a subfolder in the directory called sec-edgar-filings from where you can find the reports that it downloaded.
- The stock selection symbol is done within the main.py program itself, where you can choose the stock symbol by typing it in, and type of report if you know it (and it is available from the official site):
- ![image](https://github.com/user-attachments/assets/9bb0d0ca-d6f9-4b7c-acc2-0a1345c6f747)


## Features
- Automated SEC filing retrieval and processing
- Sentiment analysis using fine-tuned FinBERT model
- Document sectioning and contextual analysis
- Performance benchmarking against manual analysis
- [Coming Soon] RSS feed integration for real-time updates

## Technical Stack
- Python
- FinBERT (Financial domain-specific BERT model)
- SEC-API integration
- Natural Language Processing (NLP) pipeline

## Current Status
- Initial pipeline implementation complete
- Benchmark testing in progress
- Working on RSS feed integration
- Exploring expansion into real estate and private company analysis

## Future Development
- Integration with real-time financial news feeds
- Enhanced visualization of sentiment trends
- Customizable analysis parameters
- API endpoint development for third-party integration

## Usage
- Clone the repo and run the file. Will segment into modular files at a later date.

## Contributing
This project is currently in development. Feedback and suggestions are welcome.


---
*Note: This project is part of ongoing research into financial document analysis and sentiment tracking.*
