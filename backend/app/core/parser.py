from PyPDF2 import PdfReader
from pydantic import BaseModel
from typing import List, Tuple
import json
import io
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
import hashlib
import pickle
import os
import time
import datetime
import requests
from PIL import Image
import pytesseract  
from concurrent.futures import ThreadPoolExecutor
from google.cloud.firestore_v1 import SERVER_TIMESTAMP
from google.auth.transport import requests as google_requests
from google.auth import default, compute_engine
from fastapi import HTTPException
from io import BytesIO
import nltk
nltk.download('stopwords')
nltk.download('punkt')


from app.core.logging import logger
from app.core.prompts import metadata_prompt
from app.core.database import db, storage
from app.core.llms import get_embeddings_batch, chat_response

pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSERACT_EXE')
PDF_COLLECTION_NAME = os.getenv('PDF_COLLECTION_NAME')
PDF_BUCKET_NAME = os.getenv('PDF_BUCKET_NAME')
KEYWORD_COUNT = int(os.getenv('KEYWORD_COUNT'))

CHUNK_SIZE = int(os.getenv('CHUNK_SIZE'))
OVERLAP = int(os.getenv('OVERLAP'))
METADATA_NUMBER_OF_CHUNKS = int(os.getenv('METADATA_NUMBER_OF_CHUNKS'))
FILTER_PIXEL_AMOUNT = int(os.getenv('FILTER_PIXEL_AMOUNT'))

SUMMARIZER_URL = os.getenv("SUMMARIZER_URL")

async def parse_pdf(pdf_path:str):
    try:
        with open(pdf_path, 'rb') as file:
            pdf_bytes = BytesIO(file.read())
    except:
        raise HTTPException(status_code=500, detail="Unsupported file type.")
    try:
        metadata = load(pdf_bytes)
    except Exception as e:
        metadata, _, _ = generate_pdf(pdf_bytes, pdf_path)

    if 'embedded_summary' in metadata:
        del metadata['embedded_summary']
    if 'date_added' in metadata:
        del metadata['date_added']

    return metadata

class TextChunk(BaseModel):
    text: str
    start_page: int
    end_page: int

def generate_signed_url(doc_id, expiration):
    try:
        bucket = storage.bucket(PDF_BUCKET_NAME)
        blob = bucket.blob(doc_id + '.pdf')

        url = blob.generate_signed_url(
            version="v4",
            expiration=expiration,
            method="GET"
        )

        return url
    except Exception as e:
        # !!! temp fix because of google auth bug
        credentials, _ = default()
        auth_request = google_requests.Request()
        credentials.refresh(auth_request)
        
        signing_credentials = compute_engine.IDTokenCredentials(
            auth_request,
            "",
            service_account_email=credentials.service_account_email
        )
        url = blob.generate_signed_url(
            expiration,
            credentials=signing_credentials,
            version="v4"
        )
        return url

def process_page(page, i):
    try:
        words = page.extract_text().split()
        try:
            if len(page.images) > 0:
                for image in page.images:
                    data = image.data
                    im = Image.open(io.BytesIO(data))
                    if im.size[0] * im.size[1] < FILTER_PIXEL_AMOUNT:
                        continue
                    try:
                        # resize to square and max 512by512
                        min_dim = min(im.size)
                        im = im.crop((0, 0, min_dim, min_dim))
                        if im.size[0] > 512 or im.size[1] > 512:
                            im = im.resize((512, 512))
                        text = pytesseract.image_to_string(im, timeout=10)
                        words += text.split()
                    except RuntimeError as timeout_error:
                        logger.info(f'timeout on image on page {i}')
                        pass
        except Exception as e:
            logger.error(f"Error getting images on page {i+1}")
            logger.exception(e)
        return words
    except Exception as e:
        pass

def get_all_text(pdf:PdfReader, link:str) -> List[str]:
    try:
        futures = []
        with ThreadPoolExecutor() as executor:
            for i, _ in enumerate(pdf.pages):
                future = executor.submit(process_page, pdf.pages[i], i)
                futures.append(future)
        results = [future.result() for future in futures]
        chunks, all_words = chunk_pdf(results)
        embeddings = get_embeddings_batch([chunk.text for chunk in chunks])
        return embeddings, chunks, all_words
    except Exception as e:
        logger.error(f'Error extracting text from {link}')
        logger.error(e)
        raise HTTPException(status_code=500, detail="""Sorry, there was an error extracting the text. 
        I have taken note of this document and will fix the issue. If you want something summarized you can copy it directly into the chat.""")
    
def chunk_pdf(pages: list) -> Tuple[List[dict],int]:
        chunks: List[TextChunk] = []
        all_words = []
        words = []
        start_page = 0
        for i, word_list in enumerate(pages):
            if word_list is not None:
                words += word_list
                all_words += word_list

            while len(words) >= CHUNK_SIZE:
                chunk_words = words[:CHUNK_SIZE]
                chunks.append(TextChunk(text=" ".join(chunk_words), start_page=start_page+1, end_page=i+1))
                words = words[CHUNK_SIZE-OVERLAP:]
                # used that last bit of previous page
                if start_page != i:
                    start_page = i

        # Append the leftover words
        if len(words) > 0:
            chunks.append(TextChunk(text=" ".join(words), start_page=start_page+1, end_page=len(chunks)))

        return chunks, all_words

def get_metadata(chunks: List[TextChunk], link:str) -> dict:
    try:
        text = " ".join([chunk.text for chunk in chunks[:METADATA_NUMBER_OF_CHUNKS]])
        messages = [
                        {'role': 'system', 'content': metadata_prompt(text)},
                    ]
        response = chat_response(messages)
        metadata = json.loads(response.choices[0].message.content)
        return metadata
    except Exception as e:
        logger.error(f'Error extracting metadata {link}')
        logger.error(e)
        return {
            "author": '',
            "title": '',   
            "year": ''
        }

def get_keywords(all_words: List[str], link:str) -> List[str]:
    try:
        combined_stopwords = set(stopwords.words('english')) | set(stopwords.words('french')) | set(stopwords.words('spanish'))
        words = [word.lower() for word in word_tokenize(' '.join(all_words)) if word.isalnum() and word.lower() not in combined_stopwords]
        words = [word for word in words if not word.isnumeric()]
        word_freq = nltk.FreqDist(words)
        return [word for word, freq in word_freq.most_common(KEYWORD_COUNT)]
    except Exception as e:
        logger.error(f'Error getting keywords from {link}')
        logger.error(e)
        return []

def save(pdf_bytes: bytes, metadata:dict, chunks: List[TextChunk], embedded_chunk: List[List[float]]) -> None:
    doc_id = hashlib.sha256(pdf_bytes.getvalue()).hexdigest()
    metadata['date_added'] = SERVER_TIMESTAMP
    db.collection(PDF_COLLECTION_NAME).document(doc_id).set(metadata)
    bucket = storage.bucket(PDF_BUCKET_NAME)
    blob = bucket.blob(doc_id)
    blob.upload_from_string(
        data=json.dumps({
            'chunks': [chunk.model_dump() for chunk in chunks],
            'embeddings': embedded_chunk
        }),
        content_type='application/octet-stream'
    )
    blob = bucket.blob(doc_id + '.pdf')
    blob.upload_from_string(
        data=pickle.dumps(pdf_bytes),
        content_type='application/pdf'
    )

    return doc_id

def load(pdf_bytes: bytes) -> Tuple[dict, List[TextChunk], List[List[float]]]:
    doc_id = hashlib.sha256(pdf_bytes.getvalue()).hexdigest()
    if not db.collection(PDF_COLLECTION_NAME).document(doc_id).get().exists:
        raise Exception('Metadata in Firestore does not exist')
    if not storage.bucket(PDF_BUCKET_NAME).blob(doc_id).exists():
        raise Exception('Embeddings do not exist')
    if not storage.bucket(PDF_BUCKET_NAME).blob(doc_id + '.pdf').exists():
        raise Exception('PDF does not exist')

    try:
        metadata = db.collection(PDF_COLLECTION_NAME).document(doc_id).get().to_dict()
        metadata['url'] = generate_signed_url(doc_id, 60*60*24*7)
        expiration_date = datetime.datetime.now() + datetime.timedelta(seconds=60*60*24*7)
        expiration_date_str = expiration_date.strftime('%Y-%m-%d at %H:%M')
        metadata['url_expiration'] = expiration_date_str
    except Exception as e:
        logger.error(f'Error creating signed URL for document id: {doc_id}')
        logger.error(e)
    metadata['is_new'] = False
    return metadata

def get_summary(doc_id: str, link:str):
    try:
        doc_location = {
            "bucket_name": PDF_BUCKET_NAME,
            "blob_id": doc_id
        }
        summary_response = requests.post(SUMMARIZER_URL + '/api/v1/chunk-summary', json=doc_location)
        summary = summary_response.json()['summary']
        embedded_summary = get_embeddings_batch(summary)[0]
        db.collection(PDF_COLLECTION_NAME).document(doc_id).update({
            'summary': summary,
            'embedded_summary': embedded_summary
        })
        return summary, embedded_summary
    except Exception as e:
        logger.error(f'Error getting summary {link}')
        logger.error(e)
        return "Error getting summary", [] 

def get_signed_url(doc_id:str, link:str):
    try:
        url = generate_signed_url(doc_id, 60*60*24*7)
        expiration_date = datetime.datetime.now() + datetime.timedelta(seconds=60*60*24*7)
        return url, expiration_date.strftime('%Y-%m-%d at %H:%M')
    except Exception as e:
        logger.error(f'Error getting signed url {link}')
        logger.error(e)
        return "", ""

def generate_pdf(pdf_bytes: bytes, link:str) -> Tuple[dict, List[TextChunk], List[List[float]]]:
    logger.info(f"Document does not exist. Generating data...")
    tic = time.time()
    pdf = PdfReader(pdf_bytes)
    embedded_chunk, chunks, all_words = get_all_text(pdf, link)
    metadata = get_metadata(chunks, link)
    
    metadata['keywords'] = get_keywords(all_words, link)
    metadata['total_words'] = len(all_words)
    doc_id = save(pdf_bytes, metadata, chunks, embedded_chunk)
    metadata['summary'], metadata['embedded_summary'] = get_summary(doc_id, link)
    
    toc = time.time()
    logger.info(f"Generated data in {toc-tic} seconds.")
    metadata['parse_time'] = toc-tic
    metadata['url'], metadata['url_expiration'] = get_signed_url(doc_id, link)
    metadata['is_new'] = True
    return metadata, chunks, embedded_chunk