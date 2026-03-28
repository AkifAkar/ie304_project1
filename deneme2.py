import re

def clean_and_chunk_data(input_filename, output_filename):
    with open(input_filename, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split the raw content by the URL markers
    pages = content.split('--- SOURCE URL: ')
    clean_chunks = []

    # Define the repetitive website boilerplate to remove
    boilerplate = [
        "Industrial Engineering Summer Practice Skip to main content IE METU Türkçe Industrial Engineering Summer Practice Menu ▾ Home General Information Steps to Follow Documents/Forms SP Opportunities Previous SP Opportunites FAQ SP Committee",
        "Share Tweet",
        "Üniversiteler Mahallesi, Dumlupınar Bulvarı No:1, 06800 Çankaya/Ankara © ORTA DOĞU TEKNİK ÜNİVERSİTESİ ANKARA KAMPUSU"
    ]

    for page in pages:
        if not page.strip():
            continue

        # Separate the URL from the page text
        parts = page.split(' ---', 1)
        if len(parts) != 2:
            continue

        url = parts[0].strip()
        text = parts[1]

        # 1. THE FIX: Skip binary files that cause gibberish
        if url.endswith(('.doc', '.xls', '.pdf', '.docx', '.xlsx')):
            continue

        # 2. CLEAN THE TEXT
        # Remove the navigation and footer boilerplate
        for phrase in boilerplate:
            text = text.replace(phrase, '')

        # Remove the "Last Updated" timestamps using regex
        text = re.sub(r'Last Updated: \d{2}/\d{2}/\d{4} - \d{2}:\d{2}', '', text)

        # Clean up excessive whitespace, newlines, and tabs
        text = re.sub(r'\s+', ' ', text).strip()

        if not text:
            continue

        # 3. CREATE THE CHUNK
        # We attach the source URL to the chunk so the LLM can cite its sources later
        chunk = f"SOURCE: {url}\nCONTENT: {text}\n"
        clean_chunks.append(chunk)

    # Write the clean, chunked data to a new file
    with open(output_filename, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(clean_chunks):
            f.write(f"--- CHUNK {i+1} ---\n")
            f.write(chunk + "\n\n")

    return len(clean_chunks)

if __name__ == "__main__":
    input_file = "metu_sp_knowledge_base.txt"
    output_file = "clean_metu_sp_chunks.txt"
    
    print("Cleaning and chunking data...")
    num_chunks = clean_and_chunk_data(input_file, output_file)
    print(f"Success! Created {num_chunks} clean chunks saved to '{output_file}'.")