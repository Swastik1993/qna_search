# Import the specific libraries
from modules import BertSquad

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel

# Custom meta information for the API documentaion page. [can be removed if not required]
tags_metadata = [
    {
        "name": "hello world",
        "description": "Simple API to check whether the service is up or not",
    },
    {
        "name": "echo message",
        "description": "API for checking the service is working with a request body.",
    },
    {
        "name": "custom question",
        "description": "The group of API created for providing responses to the requested questions.",
    },
]

# Initialize fastapi application with custom meta information. [meta information can be removed if not required]
app = FastAPI(
                title="Conversational Q&A",
                description="This is the API documentation for the Conversational Q&A module which is created using FastAPI and custom OpenAPI schema",
                version="2.0.1",
                openapi_tags=tags_metadata
            )


# Data model definitions for the API usage

class TestMessage(BaseModel):
    """
    Data model for mapping the incoming request for echo_message() API
    """
    message: str


class SimpleQuestion(BaseModel):
    """
    Data model for mapping the incoming request for getAnswer() API
    """
    question: str


class DetailQuestion(BaseModel):
    """
    Data model for mapping the incoming request for getDetailAnswer() API
    """
    question: str
    doc_id: str


# FastAPI routes for the various API. 
# Please visit the Swagger UI having the API documentation at http://localhost:5000/docs for the schema details and interactive usage. 

@app.get("/", tags=["hello world"])
def hello_world(self):
    """
        A hello world test API for test purposes.

            Parameters:
                    (none)

            Returns:
                    message (dict as json): message as hello world.
    """
    message = {'message': 'Hello World!'}
    return message


@app.post("/message/", tags=["echo message"])
def echo_message(testMesg: TestMessage):
    """
        Test API for returning the contents of the request body.

            Parameters:
                    testMesg (json): request body containing the user message.
                        message (str): the user message.

            Returns:
                    resp_mesg (dict as json): response from the API containing the user message.
    """
    resp_mesg = {'response': testMesg}
    return resp_mesg


@app.post("/rich-text/", tags=["custom question"])
def getAnswer(simple_question: SimpleQuestion):
    """
        Custom API for handling the questions from the UI and returning the relevant answers for the given question.

            Parameters:
                    simple_question (json): request body containing the question of the user. 
                        question (json): the requested question of the user.

            Returns:
                    response_message (dict as json): response from the API containing the rich_text, a custom warning, and the list of answers as a result for the given question.
    """
    api = BertSquad()
    kw_list = ""
    question = simple_question.question
    rich_text, warn, result = api.searchDatabase(question, kw_list)
    response_message = {'rich_text': rich_text, 'warning': warn, 'result': result}
    fileObj = open('../results/rich_text.json', 'w')
    fileObj.writelines(response_message)
    fileObj.close()
    return response_message


@app.post("/detailed-text/", tags=["custom question"])
def getDetailAnswer(detail_question: DetailQuestion):
    """
        Custom API that will return the detailed highlighted answer based on the given question for the provided document id.

            Parameters:
                    detail_question (json): request body containing the question of the user and the the provided document id. 
                        question (str): the requested question of the user.
                        doc_id (str): the lucene document id on which a detailed answer is requested.

            Returns:
                    result (dict as json): response from the API containing the document id, title of the source and the highlighted answer the given question.
    """
    api = BertSquad()
    question = detail_question.question
    doc_id = detail_question.doc_id
    result = api.show_results(question, doc_id)
    fileObj = open('../results/detailed_text.json', 'w')
    fileObj.writelines(result)
    fileObj.close()
    return result


def custom_openapi():
    """
        Custom OpenAPI definitions for for the documentation pages. [can be removed if not required]

            Parameters:
                    (none)

            Returns:
                    openapi_schema (object): Object containing the custom definition for the title, version and other meta information.
    """
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Conversational Q&A",
        version="2.0.1",
        description="This is the API documentation for the Conversational Q&A module built using FastAPI and custom OpenAPI schema",
        routes=app.routes,
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://www.pngkit.com/png/detail/54-544622_bert-4-sesame-street-head-png.png"
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi


# View the interactive API documentation on http://localhost:5000/docs for alternative API documentation please visit http://localhost:5000/redoc
