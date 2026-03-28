import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import os

def crawl_and_download(base_url, download_dir="documents"):
    # 1. Create the documents folder if it doesn't exist
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
        print(f"Created directory: {download_dir}")

    visited = set()
    queue = [base_url]
    all_content = {}
    
    parsed_base = urlparse(base_url)
    base_netloc = parsed_base.netloc
    base_path = parsed_base.path
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    # Define the file types we want to download instead of read
    downloadable_extensions = ('.pdf', '.doc', '.docx', '.xls', '.xlsx')

    while queue:
        current_url = queue.pop(0)
        
        if current_url in visited:
            continue
            
        print(f"Crawling: {current_url}")
        
        try:
            # 2. Check if the current URL is a file we should download
            if current_url.lower().endswith(downloadable_extensions):
                # Extract just the file name from the URL (e.g., "IE300_Manual.pdf")
                filename = os.path.basename(urlparse(current_url).path)
                filepath = os.path.join(download_dir, filename)
                
                # Download and save the file in binary mode ('wb')
                print(f"  --> Downloading file: {filename}")
                file_response = requests.get(current_url, headers=headers)
                if file_response.status_code == 200:
                    with open(filepath, 'wb') as f:
                        f.write(file_response.content)
                visited.add(current_url)
                continue # Skip the HTML parsing below since it's a file
            
            # 3. If it's a normal web page, process it as usual
            response = requests.get(current_url, headers=headers, timeout=10)
            visited.add(current_url)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                text = soup.get_text(separator=' ', strip=True)
                all_content[current_url] = text
                
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    full_url = urljoin(current_url, href).split('#')[0]
                    parsed_url = urlparse(full_url)
                    
                    if parsed_url.netloc == base_netloc and parsed_url.path.startswith(base_path):
                        if full_url not in visited and full_url not in queue:
                            queue.append(full_url)
                            
            time.sleep(1)
            
        except requests.RequestException as e:
            print(f"Failed to process {current_url}: {e}")
            visited.add(current_url) 
            
    return all_content

if __name__ == "__main__":
    base_domain = "https://sp-ie.metu.edu.tr/en"
    print("Starting the crawl and download process...")
    scraped_data = crawl_and_download(base_domain)
    
    # Save the text data
    with open("metu_sp_knowledge_base.txt", "w", encoding="utf-8") as f:
        for url, content in scraped_data.items():
            f.write(f"--- SOURCE URL: {url} ---\n")
            f.write(content + "\n\n")
            
    print("Process complete!")