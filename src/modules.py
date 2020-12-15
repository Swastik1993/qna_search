# import the specific libraries
import os
import json
import numpy as np
import pandas as pd
import re
import gc
import requests
from bs4 import BeautifulSoup
import datetime
import dateutil.parser as dparser

import torch
import tensorflow as tf
import tensorflow_hub as hub
import torch
import transformers
from transformers import *
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
from transformers import BartTokenizer, BartForConditionalGeneration

import pyserini
from pyserini.search import pysearch
from IPython.core.display import display, HTML
from tqdm import tqdm
from Bio import Entrez, Medline
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# set JAVA_HOME path
os.environ["JAVA_HOME"] = "/usr/lib/jvm/jdk-11.0.2/"

# Initailize tensorflow module globally if you have GPU else comment out this part. Please check the no_gpu branch. 
def embed_useT():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 4GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
        except RuntimeError as e:
            print("Error Occured", e)
    module = '/home/swastik/Projects/Others/sentence_wise_email/module/module_useT'
    #gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.4)
    #sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
    with tf.Graph().as_default():
        sentences = tf.compat.v1.placeholder(tf.string)
        embed = hub.Module(module)
        embeddings = embed(sentences)
        session = tf.compat.v1.train.MonitoredSession()
    return lambda x: session.run(embeddings, {sentences: x})
embed_fn = embed_useT()



class BertSquad:
    """
    The main class which has all the fucntions defined for the different API calls.

        Functions:
            Below are the functions which are used for the rich-text API
                reconstructText():
                makeBERTSQuADPrediction():
                searchAbstracts():
                displayResults():
                getrecord():
                pubMedSearch():
                medrxivSearch():
                searchDatabase():
            Below are the functions which are used for the detailed-text API
                show_query():
                show_document():
                extract_scibert():
                get_result_id():
                cross_match():
                show_sections():
                highlight_paragraph():
                show_results():

    """
    # user configuration for different functionality
    USE_SUMMARY = True  # this will generate a rich-text after parsing every document
    FIND_PDFS = False # this is for dynamically searching PDF articles for documents with empty body
    SEARCH_MEDRXIV = False # this is for searching medRxiv data
    SEARCH_PUBMED = False # this is for searching PubMed data

    minDate = '2020/08/13' # last updated date for the lucene indexes
    # luceneDir = '/data/indexes/lucene-index-cord19/' # path for the lucene index data if building with docker
    luceneDir = '../data/lucene-index-cord19/' # path for the lucene index data

    torch_device = 'cuda' # since this is the no_gpu branch CUDA property cannot be set for GPU accereation

    # load the bert-large-uncased-whole-word-masking-finetuned-squad model
    QA_MODEL = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    QA_TOKENIZER = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    QA_MODEL.to(torch_device)
    QA_MODEL.eval()

    # load the facebook/bart-large-cnn model
    if USE_SUMMARY:
        SUMMARY_TOKENIZER = BartTokenizer.from_pretrained('facebook/bart-large-cnn') # used for final rich-text summary
        SUMMARY_MODEL = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
        SUMMARY_MODEL.to(torch_device)
        SUMMARY_MODEL.eval()

    # load the monologg/biobert_v1.1_pubmed model
    para_model = AutoModel.from_pretrained('monologg/biobert_v1.1_pubmed')
    para_tokenizer = AutoTokenizer.from_pretrained('monologg/biobert_v1.1_pubmed', do_lower_case=False)
    gc.collect()


    def reconstructText(self, tokens, start=0, stop=-1):
        """
        Returns the text after merging the specific tokens from start to end to form a readable sentence.

            Parameters:
                    tokens (list): list of tokens.
                    start (int): starting index position.
                    stop (int): stopping index position.

            Returns:
                    newList (list): list of small sentences.
        """
        tokens = tokens[start: stop]
        if '[SEP]' in tokens:
            sepind = tokens.index('[SEP]')
            tokens = tokens[sepind+1:]
        txt = ' '.join(tokens)
        txt = txt.replace(' ##', '')
        txt = txt.replace('##', '')
        txt = txt.strip()
        txt = " ".join(txt.split())
        txt = txt.replace(' .', '.')
        txt = txt.replace('( ', '(')
        txt = txt.replace(' )', ')')
        txt = txt.replace(' - ', '-')
        txt_list = txt.split(' , ')
        txt = ''
        nTxtL = len(txt_list)
        if nTxtL == 1:
            return txt_list[0]
        newList =[]
        for i,t in enumerate(txt_list):
            if i < nTxtL -1:
                if t[-1].isdigit() and txt_list[i+1][0].isdigit():
                    newList += [t,',']
                else:
                    newList += [t, ', ']
            else:
                newList += [t]
        return ''.join(newList)


    def makeBERTSQuADPrediction(self, document, question):
        """
        Returns all the predicted possible answers found for the question on a specific document set using the BERT model running on pytorch.

            Parameters:
                    document (str): the abstract documents on which the highlighting is required.
                    question (str): the question based on which the answers need to be found.

            Returns:
                    ans (dict): dictionary of the answers found by BERT along with their confidence scores.
        """
        nWords = len(document.split())
        input_ids_all = self.QA_TOKENIZER.encode(question, document)
        tokens_all = self.QA_TOKENIZER.convert_ids_to_tokens(input_ids_all)
        overlapFac = 1.1
        if len(input_ids_all)*overlapFac > 2048:
            nSearchWords = int(np.ceil(nWords/5))
            quarter = int(np.ceil(nWords/4))
            docSplit = document.split()
            docPieces = [' '.join(docSplit[:int(nSearchWords*overlapFac)]), 
                        ' '.join(docSplit[quarter-int(nSearchWords*overlapFac/2):quarter+int(quarter*overlapFac/2)]),
                        ' '.join(docSplit[quarter*2-int(nSearchWords*overlapFac/2):quarter*2+int(quarter*overlapFac/2)]),
                        ' '.join(docSplit[quarter*3-int(nSearchWords*overlapFac/2):quarter*3+int(quarter*overlapFac/2)]),
                        ' '.join(docSplit[-int(nSearchWords*overlapFac):])]
            input_ids = [self.QA_TOKENIZER.encode(question, dp) for dp in docPieces]
            
        elif len(input_ids_all)*overlapFac > 1536:
            nSearchWords = int(np.ceil(nWords/4))
            third = int(np.ceil(nWords/3))
            docSplit = document.split()
            docPieces = [' '.join(docSplit[:int(nSearchWords*overlapFac)]), 
                        ' '.join(docSplit[third-int(nSearchWords*overlapFac/2):third+int(nSearchWords*overlapFac/2)]),
                        ' '.join(docSplit[third*2-int(nSearchWords*overlapFac/2):third*2+int(nSearchWords*overlapFac/2)]),
                        ' '.join(docSplit[-int(nSearchWords*overlapFac):])]
            input_ids = [self.QA_TOKENIZER.encode(question, dp) for dp in docPieces]
            
        elif len(input_ids_all)*overlapFac > 1024:
            nSearchWords = int(np.ceil(nWords/3))
            middle = int(np.ceil(nWords/2))
            docSplit = document.split()
            docPieces = [' '.join(docSplit[:int(nSearchWords*overlapFac)]), 
                        ' '.join(docSplit[middle-int(nSearchWords*overlapFac/2):middle+int(nSearchWords*overlapFac/2)]),
                        ' '.join(docSplit[-int(nSearchWords*overlapFac):])]
            input_ids = [self.QA_TOKENIZER.encode(question, dp) for dp in docPieces]
        elif len(input_ids_all)*overlapFac > 512:
            nSearchWords = int(np.ceil(nWords/2))
            docSplit = document.split()
            docPieces = [' '.join(docSplit[:int(nSearchWords*overlapFac)]), ' '.join(docSplit[-int(nSearchWords*overlapFac):])]
            input_ids = [self.QA_TOKENIZER.encode(question, dp) for dp in docPieces]
        else:
            input_ids = [input_ids_all]
        absTooLong = False    
        
        answers = []
        cons = []
        for iptIds in input_ids:
            tokens = self.QA_TOKENIZER.convert_ids_to_tokens(iptIds)
            sep_index = iptIds.index(self.QA_TOKENIZER.sep_token_id)
            num_seg_a = sep_index + 1
            num_seg_b = len(iptIds) - num_seg_a
            segment_ids = [0]*num_seg_a + [1]*num_seg_b
            assert len(segment_ids) == len(iptIds)
            n_ids = len(segment_ids)

            if n_ids < 512:
                start_scores, end_scores = self.QA_MODEL(torch.tensor([iptIds]).to(self.torch_device),
                                        token_type_ids=torch.tensor([segment_ids]).to(self.torch_device))
            else:
                #this cuts off the text if its more than 512 words so it fits in model space 
                print('****** warning only considering first 512 tokens, document is '+str(nWords)+' words long.  There are '+str(n_ids)+ ' tokens')
                absTooLong = True
                start_scores, end_scores = self.QA_MODEL(torch.tensor([iptIds[:512]]).to(self.torch_device),
                                        token_type_ids=torch.tensor([segment_ids[:512]]).to(self.torch_device))
            start_scores = start_scores[:,1:-1]
            end_scores = end_scores[:,1:-1]
            answer_start = torch.argmax(start_scores)
            answer_end = torch.argmax(end_scores)
            answer = self.reconstructText(tokens, answer_start, answer_end+2)
        
            if answer.startswith('. ') or answer.startswith(', '):
                answer = answer[2:]
                
            c = start_scores[0,answer_start].item()+end_scores[0,answer_end].item()
            answers.append(answer)
            cons.append(c)
        
        maxC = max(cons)
        iMaxC = [i for i, j in enumerate(cons) if j == maxC][0]
        confidence = cons[iMaxC]
        answer = answers[iMaxC]
        
        sep_index = tokens_all.index('[SEP]')
        full_txt_tokens = tokens_all[sep_index+1:]
        
        abs_returned = self.reconstructText(full_txt_tokens)

        ans={}
        ans['answer'] = answer
        if answer.startswith('[CLS]') or answer_end.item() < sep_index or answer.endswith('[SEP]'):
            ans['confidence'] = -1000000
        else:
            ans['confidence'] = confidence
        ans['abstract_bert'] = abs_returned
        ans['abs_too_long'] = absTooLong
        return ans


    def searchAbstracts(self, hit_dictionary, question):
        """
        Returns all the possible answers found for the given question after searching through the hits found  document set.

            Parameters:
                    hit_dictionary (dict): dictionary containing all the documents that have been found as a match for the given question by searching through all the lucene indexes.
                    question (str): the question based on which the answers need to be found.

            Returns:
                    abstractResults (dict): dictionary of all the relevant results along with their BERT highlighted answers with confidence scores.
        """
        abstractResults = {}
        for k,v in tqdm(hit_dictionary.items()):
            abstract = v['abstract_full']
            if abstract:
                ans = self.makeBERTSQuADPrediction(abstract, question)
                if ans['answer']:
                    confidence = ans['confidence']
                    abstractResults[confidence]={}
                    abstractResults[confidence]['main_abstract'] = abstract
                    abstractResults[confidence]['answer'] = ans['answer']
                    abstractResults[confidence]['abstract_bert'] = ans['abstract_bert']
                    abstractResults[confidence]['idx'] = k
                    abstractResults[confidence]['abs_too_long'] = ans['abs_too_long']
                    
        cList = list(abstractResults.keys())
        if cList:
            maxScore = max(cList)
            total = 0.0
            exp_scores = []
            for c in cList:
                s = np.exp(c-maxScore)
                exp_scores.append(s)
            total = sum(exp_scores)
            for i,c in enumerate(cList):
                abstractResults[exp_scores[i]/total] = abstractResults.pop(c)
        return abstractResults


    def displayResults(self, hit_dictionary, answers, question):
        """
        Returns all the possible answers found for the given question after searching through the hits found  document set.

            Parameters:
                    hit_dictionary (dict): dictionary containing all the documents that have been found as a match for the given question by searching through all the lucene indexes.
                    answers (dict): dictionary of all the relevant results along with their BERT highlighted answers with confidence scores.
                    question (str): the question based on which the answers need to be found.

            Returns:
                    summ (str): generated BART summary based on all the answers.
                    warning_HTML (str): text for the UI. [can be replaced with literally anything. Not very important.]
                    df (dataframe converted to json): json data containing all the relevant fields necessary for the UI layer.
        """
        question_HTML = '<div font-size: 28px; padding-bottom:28px"><b>Query</b>: '+question+'</div>'
        confidence = list(answers.keys())
        confidence.sort(reverse=True)
        confidence = list(answers.keys())
        confidence.sort(reverse=True)

        for c in confidence:
            if c>0 and c <= 1 and len(answers[c]['answer']) != 0:
                if 'idx' not in  answers[c]:
                    continue
                rowData = []
                idx = answers[c]['idx']
                title = hit_dictionary[idx]['title']
                authors = hit_dictionary[idx]['authors'] + ' et al.'
                doi = '<a href="https://doi.org/'+hit_dictionary[idx]['doi']+'" target="_blank">' + title +'</a>'
                main_abstract = answers[c]['main_abstract']
                
                full_abs = answers[c]['abstract_bert']
                bert_ans = answers[c]['answer']
                
                split_abs = full_abs.split(bert_ans)
                sentance_beginning = split_abs[0][split_abs[0].rfind('.')+1:]
                if len(split_abs) == 1:
                    sentance_end_pos = len(full_abs)
                    sentance_end =''
                else:
                    sentance_end_pos = split_abs[1].find('. ')+1
                    if sentance_end_pos == 0:
                        sentance_end = split_abs[1]
                    else:
                        sentance_end = split_abs[1][:sentance_end_pos]
                    
                answers[c]['full_answer'] = sentance_beginning+bert_ans+sentance_end
                answers[c]['sentence_beginning'] = sentance_beginning
                answers[c]['sentence_end'] = sentance_end
                answers[c]['title'] = title
                answers[c]['doi'] = doi
                answers[c]['main_abstract'] = main_abstract
                if 'pdfLink' in hit_dictionary[idx]:
                    answers[c]['pdfLink'] = hit_dictionary[idx]['pdfLink']

            else:
                answers.pop(c)
        
        ## now rerank based on semantic similarity of the answers to the question
        cList = list(answers.keys())
        allAnswers = [answers[c]['full_answer'] for c in cList]

        messages = [question]+allAnswers

        encoding_matrix = embed_fn(messages)
        gc.collect()
        similarity_matrix = np.inner(encoding_matrix, encoding_matrix)
        rankings = similarity_matrix[1:, 0]

        for i, c in enumerate(cList):
            answers[rankings[i]] = answers.pop(c)
        
        ## forming a pandas dataframe
        confidence = list(answers.keys())
        confidence.sort(reverse=True)
        pandasData = []
        ranked_aswers = []
        for c in confidence:
            rowData=[]
            title = answers[c]['title']
            main_abstract = answers[c]['main_abstract']
            doi = answers[c]['doi']
            idx = answers[c]['idx']
            rowData += [idx]            
            sentance_html = '<div>' +answers[c]['sentence_beginning'] + " <font color='#08A293'>"+answers[c]['answer']+"</font> "+answers[c]['sentence_end']+'</div>'
            
            rowData += [sentance_html, c, doi, main_abstract]
            pandasData.append(rowData)
            ranked_aswers.append(' '.join([answers[c]['full_answer']]))
        
        if self.FIND_PDFS or self.SEARCH_MEDRXIV:
            pdata2 = []
            for rowData in pandasData:
                rd = rowData
                idx = rowData[0]
                if 'pdfLink' in answers[rowData[2]]:
                    rd += ['<a href="'+answers[rowData[2]]['pdfLink']+'" target="_blank">PDF Link</a>']
                elif self.FIND_PDFS:
                    if str(idx).startswith('pm_'):
                        pmid = idx[3:]
                    else:
                        try:
                            test = self.UrlReverse('https://doi.org/'+hit_dictionary[idx]['doi'])
                            if test is not None:
                                pmid = test.pmid
                            else:
                                pmid = None
                        except:
                            pmid = None
                    pdfLink = None
                    if pmid is not None:
                        try:
                            pdfLink = self.FindIt(str(pmid))
                        except:
                            pdfLink = None
                    if pdfLink is not None:
                        pdfLink = pdfLink.url

                    if pdfLink is None:

                        rd += ['Not Available']
                    else:
                        rd += ['<a href="'+pdfLink+'" target="_blank">PDF Link</a>']
                else:
                    rd += ['Not Available']
                pdata2.append(rowData)
        else:
            pdata2 = pandasData

        df = pd.DataFrame(pdata2, columns=['Lucene ID', 'BERT-SQuAD Answer with Highlights', 'Confidence', 'Title/Link', 'Abstract'])
        
        if self.USE_SUMMARY:
            allAnswersTxt = ' '.join(ranked_aswers[:6]).replace('\n','')
            answers_input_ids = self.SUMMARY_TOKENIZER.batch_encode_plus([allAnswersTxt], return_tensors='pt', max_length=1024)['input_ids'].to(self.torch_device)
            summary_ids = self.SUMMARY_MODEL.generate(answers_input_ids, num_beams=10, length_penalty=1.2, max_length=1024, min_length=64, no_repeat_ngram_size=4)

            summ = self.SUMMARY_TOKENIZER.decode(summary_ids.squeeze(), skip_special_tokens=True)
            execSum_HTML = '<div style="font-size:12px;color:#CCCC00"><b>BART Abstractive Summary:</b>: '+summ+'</div>'
            warning_HTML = '<div style="font-size:12px;padding-bottom:12px;color:#CCCC00;margin-top:1px"> Warning: This is an autogenerated summary based on semantic search of abstracts, please examine the results before accepting this conclusion. There may be scenarios in which the summary will not be able to clearly answer the question.</div>'
        
        if self.FIND_PDFS or self.SEARCH_MEDRXIV:
            df = pd.DataFrame(pdata2, columns = ['Lucene ID', 'BERT-SQuAD Answer with Highlights', 'Confidence', 'Title/Link', 'Abstract'])
        else:
            df = pd.DataFrame(pdata2, columns = ['Lucene ID', 'BERT-SQuAD Answer with Highlights', 'Confidence', 'Title/Link', 'Abstract'])
            
        return summ, warning_HTML, df.to_json(orient="records", force_ascii=True, default_handler=None)


    def getrecord(self, id, db):
        """
        Returns all the possible answers found for the given question after searching through the hits found  document set.

            Parameters:
                    id (str): id for the document which needs to be fetched.
                    db (str): the database which needs to be searched.

            Returns:
                    rec (object): returns the object of the text for the id.
        """
        handle = Entrez.efetch(db=db, id=id, rettype='Medline', retmode='text')
        rec = handle.read()
        handle.close()
        return rec


    def pubMedSearch(self, terms, db='pubmed', mindate='2019/12/01'):
        """
        Returns all the possible answers found for the given question after searching through the hits found  document set.

            Parameters:
                    terms (str): the keywords on which the data will be searched.
                    db (str): by default searches the pubmed data if required can be used for similar data search.
                    mindate (date as str): by default it is 1st December the date from which the search will be performed.

            Returns:
                    record_db (dict): the pubmed article(s) which has been found.
        """
        handle = Entrez.esearch(db = db, term = terms, retmax=10, mindate=mindate)
        record = Entrez.read(handle)
        record_db = {}
        for id in record['IdList']:
            try:
                record = self.getrecord(id, db)
                recfile = StringIO(record)
                rec = Medline.read(recfile)
                if 'AB' in rec and 'AU' in rec and 'LID' in rec and 'TI' in rec:
                    if '10.' in rec['LID'] and ' [doi]' in rec['LID']:
                        record_db['pm_'+id] = {}
                        record_db['pm_'+id]['authors'] = ' '.join(rec['AU'])
                        record_db['pm_'+id]['doi'] = '10.'+rec['LID'].split('10.')[1].split(' [doi]')[0]
                        record_db['pm_'+id]['abstract'] = rec['AB']
                        record_db['pm_'+id]['title'] = rec['TI']
            except:
                print("Problem trying to retrieve: " + str(id))
            
        return record_db
    Entrez.email = 'pubmedrecord@gmail.com' # sample email id which can be configured

   
    def medrxivSearch(self, query, n_pages=1):
        """
        Returns all the possible answers found for the given question after searching through the hits found  document set.

            Parameters:
                    query (str): the keywords on which the data will be searched.
                    n_pages (int): the number of pages of information that needs to be fetched.

            Returns:
                    results (dict): the medarxiv article(s) which has been found.
        """
        results = {}
        q = query
        for x in range(n_pages):
            PARAMS = {
                'page': x
            }
            r = requests.get('https://www.medrxiv.org/search/' + q, params = PARAMS)
            content = r.text
            page = BeautifulSoup(content, 'lxml')
            
            for entry in page.find_all("a", attrs={"class": "highwire-cite-linked-title"}):
                title = ""
                url = ""
                pubDate = ""
                journal = None
                abstract = ""
                authors = []
                database = "medRxiv"
                
                url = "https://www.medrxiv.org" + entry.get('href')
                
                request_entry = requests.get(url)
                content_entry = request_entry.text
                page_entry = BeautifulSoup(content_entry, 'lxml')
                doi = page_entry.find("span", attrs={"class": "highwire-cite-metadata-doi"}).text.split('doi.org/')[-1]

                #getting pubDate
                pubDate = page_entry.find_all("div", attrs = {"class": "pane-content"})
                pubDate = pubDate[10]
                pubDate = str(dparser.parse(pubDate, fuzzy = True))
                pubDate = datetime.datetime.strptime(pubDate, '%Y-%m-%d %H:%M:%S')
                pubDate = pubDate.strftime('%b %d %Y')
                date = pubDate.split()
                month = date[0]
                day = date[1]
                year = date[2]
                pubDate = {
                    'year': year,
                    'month': month,
                    'day': day
                }

                #getting title
                title = page_entry.find("h1", attrs={"class": "highwire-cite-title"}).text
                #getting abstract
                abstract = page_entry.find("p", attrs = {"id": "p-2"}).text.replace('\n', ' ')
                #getting authors 
                givenNames = page_entry.find_all("span", attrs={"class": "nlm-given-names"})
                surnames = page_entry.find_all("span",  attrs={"class": "nlm-surname"})
                names = list(zip(givenNames,surnames))
                for author in names:
                    name = author[0].text + ' ' + author[1].text
                    if name not in authors:
                        authors.append(name)
                
                result = {
                    'title': title,
                    'url': url,
                    'pubDate': pubDate,
                    'journal': journal,
                    'abstract': abstract,
                    'authors': authors[0],
                    'database': database,
                    'doi': doi,
                    'pdfLink': url+'.full.pdf'
                }
                results['mrx_'+result['doi'].split('/')[-1]] = result
                #break

        return results


    def searchDatabase(self, question, keywords):
        """
        Returns all the possible answers found for the given question after searching through the hits found  document set.

            Parameters:
                    question (str): the question based on which the answers need to be found.
                    keywords (list): list of additional keywords for the question to give more context for the answer to search.

            Returns:
                    displayResults(hit_dictionary, answers, question): returns the data from displayResults() call.
                        summ (str): generated BART summary based on all the answers.
                        warning_HTML (str): text for the UI. [can be replaced with literally anything. Not very important.]
                        df (dataframe converted to json): json data containing all the relevant fields necessary for the UI layer.
        """
        ## search the lucene database with a combination of the question and the keywords
        pm_kw = ''
        minDate='2019/12/01'
        k=20
        
        searcher = pysearch.SimpleSearcher(self.luceneDir)
        hits = searcher.search(question + '. ' + keywords, k=k)
        n_hits = len(hits)
        ## collect the relevant data in a hit dictionary
        hit_dictionary = {}
        for i in range(0, n_hits):
            doc_json = json.loads(hits[i].raw)
            idx = str(hits[i].docid)
            hit_dictionary[idx] = doc_json
            hit_dictionary[idx]['title'] = hits[i].lucene_document.get("title")
            hit_dictionary[idx]['authors'] = hits[i].lucene_document.get("authors")
            hit_dictionary[idx]['doi'] = hits[i].lucene_document.get("doi")
            
        titleList = [h['title'] for h in hit_dictionary.values()]
        
        # search for PubMed and medArxiv data dynamically
        if pm_kw:
            if SEARCH_PUBMED:
                new_hits = pubMedSearch(pm_kw, db='pubmed', mindate=minDate)
                for id,h in new_hits.items():
                    if h['title'] not in titleList:
                        titleList.append(h['title'])
                    hit_dictionary[id] = h
            if SEARCH_MEDRXIV:
                new_hits = medrxivSearch(pm_kw)
                for id,h in new_hits.items():
                    if h['title'] not in titleList:
                        titleList.append(h['title'])
                    hit_dictionary[id] = h
        
        ## scrub the abstracts in prep for BERT-SQuAD
        for idx,v in hit_dictionary.items():

            try:
                abs_dirty = v['abstract']
            except KeyError:
                print("Sorry! No abstract found.")
                abs_dirty = ''
                # uncomment the code if required search on body_text also. Will impact processing time

    #             if v['has_full_text'] == True:
    #                 print(v['paper_id'])
    #                 abs_dirty = v['body_text']
    #             else:
    #                 print(v.keys())
    #         abs_dirty = ''
    #         abs_dirty = v['abstract']

            # looks like the abstract value can be an empty list
            v['abstract_paragraphs'] = []
            v['abstract_full'] = ''

            if abs_dirty:
                # if it is a list, then the only entry is a dictionary where text is in 'text' key it is broken up by paragraph if it is in that form.  
                # make lists for every paragraph that is full abstract text as both could be valuable for BERT derrived QA

                if isinstance(abs_dirty, list):
                    for p in abs_dirty:
                        v['abstract_paragraphs'].append(p['text'])
                        v['abstract_full'] += p['text'] + ' \n\n'

                # in some cases the abstract can be straight up text so we can actually leave that alone
                if isinstance(abs_dirty, str):
                    v['abstract_paragraphs'].append(abs_dirty)
                    v['abstract_full'] += abs_dirty + ' \n\n'
        
        ## Search collected abstracts with BERT-SQuAD
        answers = self.searchAbstracts(hit_dictionary, question)
        ## displaying results in a nice format
        return self.displayResults(hit_dictionary, answers, question)


    def show_query(self, query):
        """
        Returns HTML format for the searched query.

            Parameters:
                    query (str): the question based on which the answers need to be found.

            Returns:
                    HTML text (str): returns the query with enclosed in some html styling.
        """
        return HTML('<br/><div font-size: 20px;'
                    'padding-bottom:12px"><b>Query</b>: ' + query + '</div>')


    def show_document(self, idx, doc):
        """
        Returns HTML format for the searched query.

            Parameters:
                    idx (str): the index of the document.
                    doc (str): the actual text of the document.

            Returns:
                    HTML text (str): returns the document with the title, authors and id enclosed in some html styling.
        """
        have_body_text = 'body_text' in json.loads(doc.raw)
        body_text = ' Full text available.' if have_body_text else ''
        return HTML('<div font-size: 18px; padding-bottom:10px">' +
                    f'<b>Document {idx}:</b> {doc.docid} ({doc.score:1.2f}) -- ' +
                    f'{doc.lucene_document.get("authors")} et al. ' +
                    f'{doc.lucene_document.get("title")}. ' +
                    f'<a href="https://doi.org/{doc.lucene_document.get("doi")}">{doc.lucene_document.get("doi")}</a>.'
                    + f'{body_text}</div>')


    def extract_scibert(self, text, tokenizer, model):
        """
        Extracts the contextualized vectors for the given text/query and abstracts from SciBERT/BioBERT model for highlighting relevant paragraphs.

            Parameters:
                    text (str): the text or query(question) for which the vectrs needs to be found.
                    tokenizer (str): the tokenizer that is going to be used.
                    model (str): the model that is going to be used.

            Returns:
                    text_ids (tensor): the ids for the words as a tensor
                    text_words (list): list of words for state vectors.
                    state (tensor): the state vectors for the text as a tensor.
        """
        text_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
        text_words = tokenizer.convert_ids_to_tokens(text_ids[0])[1:-1]
        n_chunks = int(np.ceil(float(text_ids.size(1)) / 510))
        states = []
        for ci in range(n_chunks):
            text_ids_ = text_ids[0, 1 + ci * 510:1 + (ci + 1) * 510]
            text_ids_ = torch.cat([text_ids[0, 0].unsqueeze(0), text_ids_])
            if text_ids[0, -1] != text_ids[0, -1]:
                text_ids_ = torch.cat([text_ids_, text_ids[0, -1].unsqueeze(0)])
            with torch.no_grad():
                state = model(text_ids_.unsqueeze(0))[0]
                state = state[:, 1:-1, :]
            states.append(state)
        state = torch.cat(states, axis=1)
        return text_ids, text_words, state[0]


    def get_result_id(self, query, doc_id, searcher):
        """
        Returns the document/search result that has been found for the given question and doc_id.

            Parameters:
                    query (str): the given question which for which a doc_id is searched.
                    doc_id (str): the document id on which the question is going to be searched.
                    searcher (str): the searching function that is going to be used to search the doc_id for the question.

            Returns:
                    hit (str): returns the entire document/result that has been found.
        """
        hits = searcher.search(query)
        display(self.show_query(query))
        for i, hit in enumerate(hits):
            if hit.docid == doc_id:
                display(self.show_document(i + 1, hit))
                return hit


    def cross_match(self, state1, state2):
        """
        Computes the cosine similarity matrix between the query and each paragraph and returns the state vectors.

            Parameters:
                    state1 (tensor): the query state vectors.
                    state2 (tensor): the paragraph state vectors.

            Returns:
                    sim (tensor): the tensor scores after calculating the cosine similarity metrics.
        """
        state1 = state1 / torch.sqrt((state1 ** 2).sum(1, keepdims=True))
        state2 = state2 / torch.sqrt((state2 ** 2).sum(1, keepdims=True))
        sim = (state1.unsqueeze(1) * state2.unsqueeze(0)).sum(-1)
        return sim


    def show_sections(self, section, text):
        """
        Returns HTML format for the searched query.

            Parameters:
                    section (str): the entire docuemnt portion question based on which the answers need to be found.
                    text (str): the text which will be from the section

            Returns:
                    HTML text (str): returns the text with enclosed in some html styling.
        """
        return HTML(
            '<div font-size: 18px; padding-bottom:10px; margin-left: 15px">' +
            f'<b>{section}</b> -- {text.replace(" ##", "")} </div>')


    def highlight_paragraph(self, ptext, rel_words, max_win=10):
        """
        Highlights the relevant phrases in each paragraph. Any word that had a high similarity to each of the query words is considered relevant. 
        Given these words, we highlight a window of 10 surrounding each of them.

            Parameters:
                    ptext (list): the list of words that are going to be highlighted.
                    rel_words (list): the scores of the words that will be used for highlighting.
                    max_win (int): the window size based on which the number of words will be highlighted. By default value is 10.

            Returns:
                    para (str): returns the highlighted text enclosed in some html styling.
        """
        para = ""
        prev_idx = 0
        for jj in rel_words:
            if prev_idx > jj:
                continue
            found_start = False
            for kk in range(jj, prev_idx - 1, -1):
                if ptext[kk] == "." and (ptext[kk + 1][0].isupper() or ptext[kk + 1][0] == '['):
                    sent_start = kk
                    found_start = True
                    break
            if not found_start:
                sent_start = prev_idx - 1
            found_end = False
            for kk in range(jj, len(ptext) - 1):
                if ptext[kk] == "." and (ptext[kk + 1][0].isupper() or ptext[kk + 1][0] == '['):
                    sent_end = kk
                    found_end = True
                    break
            if not found_end:
                if kk >= len(ptext) - 2:
                    sent_end = len(ptext)
                else:
                    sent_end = jj
            para = para + " "
            para = para + " ".join(ptext[prev_idx:sent_start + 1])
            para = para + " <font color='blue'>"
            para = para + " ".join(ptext[sent_start + 1:sent_end])
            para = para + "</font> "
            prev_idx = sent_end
        if prev_idx < len(ptext):
            para = para + " ".join(ptext[prev_idx:])
        return para


    def show_results(self, question, doc_id):
        """
        Returns HTML format for the searched query.

            Parameters:
                    question (str): query (str): the question based on which the answers need to be found.
                    doc_id (str): the document id on which the question is going to be searched.

            Returns:
                    data (dict): returns the id, title, metadata and text from the document in a dictionary.
        """
        searcher = pysearch.SimpleSearcher(self.luceneDir)
        query = (question)
        highlighted_text = ""
        query_ids, query_words, query_state = self.extract_scibert(query, self.para_tokenizer, self.para_model)
        req_doc = json.loads(self.get_result_id(query, doc_id, searcher).raw)
        paragraph_states = []
        for par in tqdm(req_doc['body_text']):
            state = self.extract_scibert(par['text'], self.para_tokenizer, self.para_model)
            paragraph_states.append(state)
        sim_matrices = []
        for pid, par in tqdm(enumerate(req_doc['body_text'])):
            sim_score = self.cross_match(query_state, paragraph_states[pid][-1])
            sim_matrices.append(sim_score)
        paragraph_relevance = [torch.max(sim).item() for sim in sim_matrices]

        # Select the index of top 5 paragraphs with highest relevance
        rel_index = np.argsort(paragraph_relevance)[-5:][::-1]
        for ri in np.sort(rel_index):
            sim = sim_matrices[ri].data.numpy()

            # Select the two highest scoring words in the paragraph
            rel_words = np.sort(np.argsort(sim.max(0))[-2:][::-1])
            p_tokens = paragraph_states[ri][1]
            para = self.highlight_paragraph(p_tokens, rel_words)
            highlighted_text += para
            display(self.show_sections(req_doc["body_text"][ri]['section'], para))
        data = {'id': doc_id, 'title': req_doc['metadata']['title'], 'text': highlighted_text}
        return data
