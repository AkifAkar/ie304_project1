import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time

def crawl_sp_website(base_url):
    # Keep track of where we've been and where we need to go
    visited = set()
    queue = [base_url]
    all_content = {}
    
    # Parse the base URL to ensure we stay within the target domain
    parsed_base = urlparse(base_url)
    base_netloc = parsed_base.netloc
    base_path = parsed_base.path
    
    # Use a standard User-Agent to prevent the server from rejecting the request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    while queue:
        current_url = queue.pop(0)
        
        if current_url in visited:
            continue
            
        print(f"Crawling: {current_url}")
        
        try:
            response = requests.get(current_url, headers=headers, timeout=10)
            visited.add(current_url)
            
            # Only process successful page loads
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract the visible text, stripping away HTML tags
                text = soup.get_text(separator=' ', strip=True)
                all_content[current_url] = text
                
                # Find all hyperlinks on the current page
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    full_url = urljoin(current_url, href)
                    
                    # Normalize the URL by stripping off any anchor tags (#)
                    full_url = full_url.split('#')[0]
                    parsed_url = urlparse(full_url)
                    
                    # Ensure the link belongs to the target METU-IE SP domain
                    if parsed_url.netloc == base_netloc and parsed_url.path.startswith(base_path):
                        if full_url not in visited and full_url not in queue:
                            queue.append(full_url)
                            
            # Polite delay so we don't overwhelm the METU servers
            time.sleep(1)
            
        except requests.RequestException as e:
            print(f"Failed to fetch {current_url}: {e}")
            # Add to visited even on failure to prevent continuous retries
            visited.add(current_url) 
            
    return all_content

if __name__ == "__main__":
    # The primary knowledge base domain
    base_domain = "https://sp-ie.metu.edu.tr/en"
    print("Starting the crawl...")
    scraped_data = crawl_sp_website(base_domain)
    
    # Save the aggregated text data to a single file
    output_filename = "metu_sp_knowledge_base.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        for url, content in scraped_data.items():
            f.write(f"--- SOURCE URL: {url} ---\n")
            f.write(content + "\n\n")
            
    print(f"\nScraping complete. Crawled {len(scraped_data)} pages.")
    print(f"Data saved to {output_filename}")