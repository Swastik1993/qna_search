# Functions serving Rich Answers

!!! note
    The main application is broken down into two sections. One is for the rich-text API that gives all the results along with a brief summary of them and another is the detailed-text API that gives the answer from a specific paper but a much more detailed answer not just a summary. 
    This page will define all the function definition that are used for serving the `detailed-text` answers. The docstring for every function is also present in the code.

These functions are defined in the `BertSquad` class in `modules.py` file for serving the rich-text API.

```
class BertSquad:
    """
        The main class which has all the fucntions defined for serving the different API calls.
        Functions:
            show_query():
            show_document():
            extract_scibert():
            get_result_id():
            cross_match():
            show_sections():
            highlight_paragraph():
            show_results():
    """
```
___

## show_query()
```
def show_query(self, query):
    """
    Returns HTML format for the searched query.

        Parameters:
                query (str): the question based on which the answers need to be found.

        Returns:
                HTML text (str): returns the query with enclosed in some html styling.
    """
```
___

## show_document()
```
def show_document(self, idx, doc):
    """
    Returns HTML format for the searched query.

        Parameters:
                idx (str): the index of the document.
                doc (str): the actual text of the document.

        Returns:
                HTML text (str): returns the document with the title, authors and id enclosed in some html styling.
    """
```
___

## extract_scibert()
```
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
```
___

## get_result_id()
```
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
```
___

## cross_match()
```
def cross_match(self, state1, state2):
    """
    Computes the cosine similarity matrix between the query and each paragraph and returns the state vectors.

        Parameters:
                state1 (tensor): the query state vectors.
                state2 (tensor): the paragraph state vectors.

        Returns:
                sim (tensor): the tensor scores after calculating the cosine similarity metrics.
    """
```
___

## show_sections()
```
def show_sections(self, section, text):
    """
    Returns HTML format for the searched query.

        Parameters:
                section (str): the entire docuemnt portion question based on which the answers need to be found.
                text (str): the text which will be from the section

        Returns:
                HTML text (str): returns the text with enclosed in some html styling.
    """
```
___

## highlight_paragraph()
```
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
```
___

## show_results()
```
def show_results(self, question, doc_id):
    """
    Returns HTML format for the searched query.

        Parameters:
                question (str): query (str): the question based on which the answers need to be found.
                doc_id (str): the document id on which the question is going to be searched.

        Returns:
                data (dict): returns the id, title, metadata and text from the document in a dictionary.
    """
```
___
