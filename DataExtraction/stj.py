
import csv
from Utilities import chromeBrowser, download_pdf, read_pdf_from_file
from timeit import default_timer as timer
import os
import re
from multiprocessing import Pool, Array, Value
import multiprocessing
from time import sleep


start_time = timer()

error = None
current_documents = None
pool = None

output_path = os.path.expanduser('~/Desktop/AI/Jurisprudence/STJ')

def error_message(message):
    thread_id = int(multiprocessing.current_process().name.split('-')[1])
    return f'(Error - Thread ID: {thread_id}) {message}'

def get_pdf(browser, document_div, document_number):
    # document_div.find_by_css('#acoesdocumento a')[0].click()
    document_div.find_by_id('acoesdocumento').find_by_text('Acórdão').click()
    browser.windows.current.next.is_current = True
    
    if len(browser.find_by_css('.tabelaacordaos a')) > 0:
        browser.find_by_css('.tabelaacordaos a')[0].click()
        browser.windows.current.next.is_current = True
    
    link = browser.find_by_css('.arvore_documentos a')[0]
    pdf_name = link.text.replace('/', '_')
    link.click()
    browser.windows.current.next.is_current = True
    
    pdf_url = browser.find_by_css('a')['href']
    
    pdf_file_path = f"{output_path}/PDFs/{document_number}-{pdf_name}.pdf"
    try:
        download_pdf(from_url = pdf_url, to_file_path = pdf_file_path)
    except Exception as e:
        print(f'(Error) URL: {pdf_url} - File path: {pdf_file_path}')
        raise e
    
    browser.windows[0].close_others()
    browser.windows[0].close_others()
    
    return pdf_name, read_pdf_from_file(pdf_file_path)

def extract_page(browser, writer):
    for document_div in browser.find_by_css('#listadocumentos > div'):
        document_number = document_div.find_by_css('h3 a').text
        print(f'{document_number} - Time: {int(timer() - start_time)}s')
        
        content = document_div.find_by_css('.paragrafoBRS .docTexto')
        
        try:
            pdf_name, pdf_text = get_pdf(browser, document_div, document_number)
        except Exception as e:
            print(f'(Error) Document: {document_number}')
            raise e
        
        writer.writerow({
            'document_number': document_number,
            'process': content[0].text,
            'reporter': content[1].text,
            'judging_body': content[2].text,
            'judgment_date': content[3].text,
            'publication_date': content[4].text,
            'summary': content[5].text,
            'judgment': content[6].text,
            'pdf_name': pdf_name,
            'pdf_text': pdf_text
        })

def thread_function(first_document_number):
    global current_documents#, error
    
    # if error.value == 1:
    #     print(error_message('Stopping!'))
    #     sleep(24 * 60 * 60)
    
    thread_id = int(multiprocessing.current_process().name.split('-')[1])
    current_documents[thread_id - 1] = first_document_number
    
    print(f'Thread ID: {thread_id} - Documents: {list(current_documents)} - Time: {int(timer() - start_time)}s')
    
    if thread_id == 1:
        sleep(1)
    
    l = list(filter(lambda x: x != 0, current_documents))
    delta = max(l) - min(l)
    if l and delta >= 200:
        print(error_message(f'Some thread got stuck! {min(list(current_documents))}'))
        sleep(24 * 60 * 60)
        raise Exception('Some thread got stuck!')
    
    browser = chromeBrowser(headless = True)
    browser.visit('https://scon.stj.jus.br/SCON/pesquisar.jsp')
    browser.find_by_name('livre')[1].fill('acórdão')
    browser.find_by_value('Pesquisar').click()
    onclick_string = f"javascript:navegaForm('{first_document_number}');"
    browser.execute_script(f"$('#navegacao .iconeProximaPagina')[0].setAttribute('onclick', \"{onclick_string}\")")
    browser.find_by_css('#navegacao .iconeProximaPagina').click()
    
    with open(f'{output_path}/STJ.csv', 'a', newline = '') as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames = [
                'document_number',
                'process',
                'reporter',
                'judging_body',
                'judgment_date',
                'publication_date',
                'summary',
                'judgment',
                'pdf_name',
                'pdf_text'
            ]
        )
        
        try:
            extract_page(browser, writer)
        except Exception as e:
            print(error_message(f'First document numbber: {first_document_number}'))
            print(error_message(str(e)))
            
            try:
                print(error_message('Second try'))
                sleep(10)
                extract_page(browser, writer)
            except Exception as e2:
                print(error_message('Second try failed'))
                print(error_message(str(e2)))
                
                # with error.get_lock():
                #     error.value = 1
                # sleep(24 * 60 * 60)
                # raise e2

if __name__ == '__main__':
    number_of_threads = 4
    
    error = Value('i', 0)
    current_documents = Array('i', [0] * number_of_threads)
    
    pool = Pool(processes = number_of_threads)
    
    pool.map(thread_function, range(681700, 683364 + 1, 10), chunksize = 1)
    # pool.map(thread_function, range(1, 200 + 1, 10), chunksize = 1)
    
    pool.close()
    
    print(f'Finished - Time: {int(timer() - start_time)}s')
