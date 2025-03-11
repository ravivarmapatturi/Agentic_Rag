import requests
from bs4 import BeautifulSoup
import os
import urllib.parse
import shutil
def download_pdfs_from_fda(url, output_folder="./data/fda_pdfs"):
    """
    Downloads PDF files from a given FDA CDER MAPP webpage.

    Args:
        url (str): The URL of the FDA CDER MAPP page.
        output_folder (str): The folder to save the downloaded PDFs.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        soup = BeautifulSoup(response.content, "html.parser")

        pdf_links = []
        file_links = []
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            full_url = urllib.parse.urljoin(url, href)
            # data_entity = a_tag["data-entity-substitution"]
            if href.lower().endswith(".pdf"):
                
                pdf_links.append(full_url)
                
            if href.startswith("/media/"):
                # print(a_tag)
                for attr, value in a_tag.attrs.items():
                    # print(f"{attr}: {value}")
                    
                    if attr =="title":
                        name=value.split("/")[-1]
                        # print(name)
                file_links.append((name,full_url))
                
            # print(data_entity)
            

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for pdf_url in pdf_links:
            try:
                pdf_response = requests.get(pdf_url, stream=True)
                pdf_response.raise_for_status()

                filename = os.path.join(output_folder, pdf_url.split("/")[-1])

                with open(filename, "wb") as pdf_file:
                    for chunk in pdf_response.iter_content(chunk_size=8192):
                        pdf_file.write(chunk)

                print(f"Downloaded: {filename}")

            except requests.exceptions.RequestException as pdf_err:
                print(f"Error downloading {pdf_url}: {pdf_err}")
        
        print(len(file_links))
        for name,file_url in file_links:
            try:
                file_response = requests.get(file_url, stream=True)
                file_response.raise_for_status()

                # filename = os.path.join(output_folder, file_url.split("/")[-1])
                filename=os.path.join(output_folder,name+".pdf")
                

                with open(filename, "wb") as file:
                    for chunk in file_response.iter_content(chunk_size=8192):
                        # print(f"Chunk received: {chunk[:50]}...")
                        file.write(chunk)
        

                print(f"Downloaded: {filename}")

            except requests.exceptions.RequestException as file_err:
                print(f"Error downloading {file_url}: {file_err}")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching the webpage: {e}")
    except Exception as general_e:
        print(f"An unexpected error occurred: {general_e}")
        
        
    return output_folder


# Example usage:
# fda_url = "https://www.fda.gov/about-fda/center-drug-evaluation-and-research-cder/cder-manual-policies-procedures-mapp"
# download_pdfs_from_fda(fda_url)