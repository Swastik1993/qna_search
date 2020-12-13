# Functions serving Rich Answers

!!! note
    The main application is broken down into two sections. One is for the rich-text API that gives all the results along with a brief summary of them and another is the detailed-text API that gives the answer from a specific paper but a much more detailed answer not just a summary. 
    This page will define all the function definition that are used for serving the `rich-text` answers. The docstring for every function is also present in the code.

These functions are defined in the `BertSquad` class in `modules.py` file for serving the rich-text API.

```
class BertSquad:
    """
        The main class which has all the fucntions defined for serving the different API calls.
        Functions:
            reconstructText():
            makeBERTSQuADPrediction():
            searchAbstracts():
            displayResults():
            getrecord():
            pubMedSearch():
            medrxivSearch():
            searchDatabase():
    """
```
___

## reconstructText()
```
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
```
___

## makeBERTSQuADPrediction()

```
def makeBERTSQuADPrediction(self, document, question):
    """
    Returns all the predicted possible answers found for the question on a specific document set using the BERT model running on pytorch.

        Parameters:
            document (str): the abstract documents on which the highlighting is required.
            question (str): the question based on which the answers need to be found.

        Returns:
            ans (dict): dictionary of the answers found by BERT along with their confidence scores.
    """
```
___

## searchAbstracts()
```
def searchAbstracts(self, hit_dictionary, question):
    """
    Returns all the possible answers found for the given question after searching through the hits found  document set.

        Parameters:
            hit_dictionary (dict): dictionary containing all the documents that have been found as a match for the given question by searching through all the lucene indexes.
            question (str): the question based on which the answers need to be found.

        Returns:
            abstractResults (dict): dictionary of all the relevant results along with their BERT highlighted answers with confidence scores.
    """
```
___

## displayResults()
```
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
```
___

## getrecord()
```
def getrecord(self, id, db):
    """
    Returns all the possible answers found for the given question after searching through the hits found  document set.

        Parameters:
            id (str): id for the document which needs to be fetched.
            db (str): the database which needs to be searched.

        Returns:
            rec (object): returns the object of the text for the id.
    """
```
___

## pubMedSearch()
```
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
```
___

## medrxivSearch()
```
def medrxivSearch(self, query, n_pages=1):
    """
    Returns all the possible answers found for the given question after searching through the hits found  document set.

        Parameters:
            query (str): the keywords on which the data will be searched.
            n_pages (int): the number of pages of information that needs to be fetched.

        Returns:
            results (dict): the medarxiv article(s) which has been found.
    """
```
___

## searchDatabase()
```
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
```
___
