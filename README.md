# NASA Knowledge Base Chatbot

An AI-powered chatbot that provides answers to questions about space, physics, and NASA's work using a comprehensive knowledge base of NASA documents, space science books, and physics literature.

## Features

- 🤖 Powered by Groq's Llama3-8b model for fast and accurate responses
- 📚 Knowledge base includes:
  - NASA official documents and technical reports
  - Space science and astronomy books
  - Physics and cosmology literature
- 🔍 Uses RAG (Retrieval-Augmented Generation) for accurate, source-based answers
- 💾 Persistent vector database for efficient document retrieval
- 📝 Simple command-line interface for easy interaction

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/nasa-chatbot.git
   cd nasa-chatbot
   ```

2. **Set up a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your environment variables**
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` and add your Groq API key:
   ```
   GROQ_API_KEY=your_api_key_here
   ```

5. **Download and process the knowledge base**
   ```bash
   python data_processing/ingest.py
   ```

6. **Start the chatbot**
   ```bash
   python qa_cli.py
   ```

## Usage

Once the chatbot is running, you can ask questions about:
- NASA missions and space exploration
- Physics and cosmology concepts
- Space science and astronomy
- Technical aspects of space technology

Example questions:
- "What are NASA's analog missions and why are they important?"
- "How does the human body adapt to space conditions?"
- "What is string theory according to Brian Greene?"
- "How did Stephen Hawking explain black holes?"

Type 'exit' to quit the chatbot.

## Knowledge Base

The chatbot's knowledge comes from:
1. NASA Official Documents
   - Technical reports
   - Mission documents
   - Research papers
2. Space Science Books
   - Works by Carl Sagan
   - Neil deGrasse Tyson's books
   - Astronomy guides
3. Physics and Cosmology Literature
   - Stephen Hawking's works
   - Brian Greene's books
   - Kip Thorne's publications

## Technical Details

- Uses LangChain for RAG implementation
- ChromaDB for vector storage
- HuggingFace's all-MiniLM-L6-v2 for embeddings
- Groq's Llama3-8b for language model

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NASA for their public documents and APIs
- Groq for providing the language model
- All the authors whose works are included in the knowledge base
